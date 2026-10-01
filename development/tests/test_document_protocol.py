import copy
import json
import pytest
from backend.preprocessing.documents import ingest_document
from backend.agents.native import NativePlanner, validate_action
from backend.agents.document_protocol import tool_specs
from backend.agents.document_agent import run_document_agent
from backend.agents.llm import LLM, LLMCallLog

@pytest.mark.parametrize('action',[
    {'tool':'read_document','parameters':{'document_id':'outside'}},
    {'tool':'read_document','parameters':{'document_id':'selected','limit':True}},
    {'tool':'answer','parameters':{'answer':'invented'}},
    {'tool':'table_tool','parameters':{'document_id':'selected','name':'analyze_data','parameters':{}}},
    {'tool':'search_documents','parameters':{'query':'rate','document_ids':[]}},
])
def test_validation_rejects_scope_types_and_extra_fields(action):
    with pytest.raises(ValueError): validate_action(action,tool_specs(['selected']))


def test_text_tool_requests_never_become_evidence(native_script):
    doc=ingest_document('rates.txt',b'Rate USD 88.')
    requests=native_script(['{"answer":"Rate USD 999."}',
        {'tool':'read_document','parameters':{'document_id':doc.id}},
        {'tool':'answer','parameters':{}},'USD 88 [E1].'])
    r=run_document_agent({doc.id:doc},[doc.id],'Rate?')
    assert r['protocol']['repair_attempts']==1
    assert '999' not in json.dumps(r['evidence'])
    assert 'native tool calls' in requests[1]['messages'][-1]['content'][0]['text']


def test_repair_budget_is_bounded(native_script):
    doc=ingest_document('rates.txt',b'Rate USD 88.')
    requests=native_script(['{"tool":"answer","parameters":{}}']*3+['USD 88 [E1].'])
    r=run_document_agent({doc.id:doc},[doc.id],'Rate?')
    # Three planner turns, then the writer; only the automatic read of the short document is evidence.
    assert len(requests)==4 and [e.get('auto') for e in r['evidence']]==[True]
    assert r['protocol']['validation_errors']==3
    assert r['limitations']


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
    planner=NativePlanner(Fake(),'system',{},tool_specs(['selected']))
    with pytest.raises(ValueError,match='max_tokens'): planner.request()
    planner.feedback([],error='Retry a complete request')
    assert planner.messages[-1]['content'][0]['toolResult']['toolUseId']=='a'


def test_native_log_snapshots_are_immutable():
    class Client:
        def converse(self,**kwargs):
            assert kwargs['inferenceConfig']['maxTokens']==100
            assert 'toolConfig' in kwargs
            return response([{'toolUseId':'a','name':'answer','input':{}}])
    log=LLMCallLog()
    llm=LLM(model_id='test',params={'maxTokens':100},call_log=log)
    llm._client=Client()
    messages=[{'role':'user','content':[{'text':'Question'}]}]
    llm.converse('system',messages,{'tools':tool_specs(['selected'])})
    messages.append({'role':'assistant','content':[{'text':'later'}]})
    assert len(log.records[0]['messages'])==1
    assert log.records[0]['stop_reason']=='tool_use'


@pytest.mark.parametrize('model',['maverick','claude-haiku-4.5','claude-sonnet-4.5','claude-opus-4.5'])
def test_every_model_uses_native_even_with_obsolete_json_setting(monkeypatch,native_script,model):
    from backend.config.settings import MODELS
    monkeypatch.setenv('DOCUMENTS_PLANNER_MODE','json')
    monkeypatch.delenv('BEDROCK_MODEL_ID',raising=False)
    doc=ingest_document('rates.txt',b'Rate USD 88.')
    requests=native_script([{'tool':'read_document','parameters':{'document_id':doc.id}},
                           {'tool':'answer','parameters':{}},'USD 88 [E1].'])
    result=run_document_agent({doc.id:doc},[doc.id],'Rate?',model=model)
    assert result['protocol']['planner_mode']=='native'
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
