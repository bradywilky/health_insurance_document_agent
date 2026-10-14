import copy
import json
import pytest
from backend.preprocessing.documents import ingest_document
from backend.agents.turn_session import TurnSession, validate_action
from backend.agents.document_protocol import tool_specs
from backend.agents.document_agent import run_document_agent, MAX_PROSE_REPLIES
from backend.agents.llm import LLM, Usage
from development.observability import capture

@pytest.mark.parametrize('action',[
    {'tool':'read_document','parameters':{'document_id':'outside'}},
    {'tool':'read_document','parameters':{'document_id':'selected','limit':True}},
    {'tool':'answer','parameters':{'answer':'invented'}},
    {'tool':'table_tool','parameters':{'document_id':'selected','name':'analyze_data','parameters':{}}},
    {'tool':'search','parameters':{'query':'rate','document_ids':['selected']}},
])
def test_validation_rejects_scope_types_and_extra_fields(action):
    with pytest.raises(ValueError): validate_action(action,tool_specs(['selected']))


def test_text_tool_requests_never_become_evidence(native_script):
    doc=ingest_document('rates.txt',b'Rate USD 88.')
    requests=native_script(['{"answer":"Rate USD 999."}',
        {'tool':'read_document','parameters':{'document_id':doc.id}},
        {'tool':'answer','parameters':{}},'USD 88 [E1].'])
    r=run_document_agent({doc.id:doc},[doc.id],'Rate?')
    # A text reply is nudged back to tool calls without spending a repair attempt.
    assert r['protocol']['prose_replies']==1 and r['protocol']['repair_attempts']==0
    assert '999' not in json.dumps(r['evidence'])
    assert 'native tool call' in requests[1]['messages'][-1]['content'][0]['text']


def test_repair_budget_is_bounded(native_script):
    doc=ingest_document('rates.txt',b'Rate USD 88.')
    requests=native_script([{'tool':'read_document','parameters':{'document_id':'outside'}}]*3+['USD 88 [E1].'])
    r=run_document_agent({doc.id:doc},[doc.id],'Rate?')
    # Three invalid planner turns, then the writer; only the automatic read of the short document is evidence.
    assert len(requests)==4 and [e.get('auto') for e in r['evidence']]==[True]
    assert r['protocol']['validation_errors']==3
    assert r['limitations']


def test_text_replies_are_bounded_apart_from_repairs(native_script):
    doc=ingest_document('rates.txt',b'Rate USD 88.')
    requests=native_script(['Let me read the rates next.']*MAX_PROSE_REPLIES+['USD 88 [E1].'])
    r=run_document_agent({doc.id:doc},[doc.id],'Rate?')
    assert len(requests)==MAX_PROSE_REPLIES+1 and [e.get('auto') for e in r['evidence']]==[True]
    assert r['protocol']['prose_replies']==MAX_PROSE_REPLIES and r['protocol']['validation_errors']==0
    assert any('stopped making tool calls' in l for l in r['limitations'])


def test_narration_between_tool_calls_does_not_end_research(native_script):
    doc=ingest_document('rates.txt',b'Rate USD 88.')
    narrate=['I will read the file next.']*(MAX_PROSE_REPLIES-1)
    native_script(narrate+[{'tool':'read_document','parameters':{'document_id':doc.id}}]+narrate
                  +[{'tool':'answer','parameters':{}},'Rate USD 88.'])
    r=run_document_agent({doc.id:doc},[doc.id],'Rate?')
    assert r['protocol']['prose_replies']==2*(MAX_PROSE_REPLIES-1)
    assert not r['limitations'] and not any(e.get('auto') for e in r['evidence'])


def response(calls,stop='tool_use'):
    return {'stopReason':stop,'usage':{'inputTokens':2,'outputTokens':1},
            'output':{'message':{'role':'assistant','content':[{'toolUse':c} for c in calls]}}}


def test_native_multiple_calls_duplicate_cache_and_result_ids(monkeypatch):
    doc=ingest_document('rates.txt',b'Rate USD 88.')
    inputs=[]
    replies=iter([response([
        {'toolUseId':'a','name':'read_document','input':{'document_id':doc.id}},
        {'toolUseId':'b','name':'read_document','input':{'document_id':doc.id}},
        {'toolUseId':'c','name':'read_document','input':{'document_id':'outside'}}]),
        response([{'toolUseId':'d','name':'answer','input':{}}])])
    def converse(self,system,messages,tool_config):
        inputs.append(copy.deepcopy(messages))
        return next(replies)
    monkeypatch.setattr(LLM,'converse',converse)
    monkeypatch.setattr('backend.agents.document_agent._call',lambda *a,**kw:'USD 88 [E1, rates.txt].')
    r=run_document_agent({doc.id:doc},[doc.id],'Rate?')
    results=inputs[1][-1]['content']
    assert [b['toolResult']['toolUseId'] for b in results]==['a','b','c']
    assert 'already_retrieved' in results[1]['toolResult']['content'][0]['text']
    assert 'error' in results[2]['toolResult']['content'][0]['text']
    assert len(r['evidence'])==1
    assert r['protocol']['repair_attempts']==1
    assert not r['limitations']


def test_truncated_native_response_is_not_executed():
    class Fake:
        def converse(self,*a):
            return response([{'toolUseId':'a','name':'answer','input':{}}],stop='max_tokens')
    session=TurnSession(Fake(),'system',{},tool_specs(['selected']))
    with pytest.raises(ValueError,match='max_tokens'): session.request()
    session.feedback([],error='Retry a complete request')
    assert session.messages[-1]['content'][0]['toolResult']['toolUseId']=='a'


def test_llm_span_records_the_call_as_sent(monkeypatch):
    monkeypatch.setenv('TRACE_CONTENT','true')
    class Client:
        def converse(self,**kwargs):
            assert kwargs['inferenceConfig']['maxTokens']==100
            assert 'toolConfig' in kwargs
            return response([{'toolUseId':'a','name':'answer','input':{}}])
    usage=Usage()
    llm=LLM(model_id='test',params={'maxTokens':100},usage_meter=usage)
    llm._client=Client()
    messages=[{'role':'user','content':[{'text':'Question'}]}]
    with capture.recording() as events:
        llm.converse('system',messages,{'tools':tool_specs(['selected'])})
    messages.append({'role':'assistant','content':[{'text':'later'}]})
    [call]=capture.summarize(events)['calls']
    assert [m['role'] for m in call['input']]==['system','user'] and call['input'][1]['content']=='Question'
    assert call['stop_reason']=='tool_use' and call['tool_calls']==['answer']
    assert call['output'][0]['tool_calls'][0]['tool_id']=='a'
    assert usage.totals()=={'llm_calls':1,'input_tokens':2,'output_tokens':1}


@pytest.mark.parametrize('model',['maverick','claude-haiku-4.5','claude-sonnet-4.5','claude-opus-4.5'])
def test_every_model_uses_native_tools(monkeypatch,native_script,model):
    from backend.config.settings import MODELS
    monkeypatch.delenv('BEDROCK_MODEL_ID',raising=False)
    doc=ingest_document('rates.txt',b'Rate USD 88.')
    requests=native_script([{'tool':'read_document','parameters':{'document_id':doc.id}},
                           {'tool':'answer','parameters':{}},'USD 88 [E1].'])
    result=run_document_agent({doc.id:doc},[doc.id],'Rate?',model=model)
    assert result['status']=='answered'
    assert all(r['modelId']==MODELS[model] for r in requests)
    assert 'toolConfig' in requests[0] and 'toolConfig' in requests[1]


def test_unsupported_native_api_does_not_fall_back(monkeypatch):
    from botocore.exceptions import ClientError
    calls=[]
    def fail(self,*args):
        calls.append(1)
        raise ClientError({'Error':{'Code':'ValidationException','Message':'Tool use not supported'}},'Converse')
    monkeypatch.setattr(LLM,'converse',fail)
    doc=ingest_document('rates.txt',b'USD 88')
    with pytest.raises(ClientError,match='not supported'):
        run_document_agent({doc.id:doc},[doc.id],'Rate?')
    assert len(calls)==1


def test_single_document_tools_and_prompt_drop_multi_document_features():
    from backend.agents.document_agent import system_prompt, writer_prompt
    single = {s['toolSpec']['name']: s['toolSpec'] for s in tool_specs(['only'])}
    several = {s['toolSpec']['name']: s['toolSpec'] for s in tool_specs(['a', 'b'])}
    assert 'list_documents' not in single and 'list_documents' in several
    assert 'document_ids' not in single['search']['inputSchema']['json']['properties']
    assert 'document_ids' in several['search']['inputSchema']['json']['properties']
    with pytest.raises(ValueError):
        validate_action({'tool':'search','parameters':{'query':'rate','document_ids':['only']}}, tool_specs(['only']))
    assert 'list_documents' not in system_prompt(True) and 'document_ids' not in system_prompt(True)
    assert 'list_documents' in system_prompt(False) and 'priority among documents' in writer_prompt(False)
    assert 'priority among documents' not in writer_prompt(True)
