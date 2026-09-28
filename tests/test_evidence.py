from pathlib import Path
import json
from unittest.mock import Mock

import table_agent as agent
from local_data import prepare_local_file
from evidence import planner_messages




def test_initial_prompt_omits_profiles_and_source_row_arrays(insurance_workbook):
    state = agent.node_init({**prepare_local_file(insurance_workbook), 'query':'What is this workbook for?'})
    assert 'top_values' not in state['system_prompt']
    assert 'source_rows' not in state['system_prompt']
    assert 'Meadow Mapping' in state['system_prompt']


def test_graph_uses_query_and_structured_evidence(monkeypatch, insurance_workbook):
    decisions = iter([
        {'tool':'query_table','parameters':{'sheet_name':'Claim Samples',
            'aggregations':[{'op':'count_rows','as':'claim_count'}]}},
        {'tool':'answer','parameters':{}},
    ])
    monkeypatch.setattr(agent, '_call_llm', lambda *a, **kw: (next(decisions), {}))
    def synth(query, findings, **kw):
        payload = json.loads(findings.split('\n\n', 1)[1])
        record = payload['evidence'][0]
        assert record['id'] == 'E1'
        assert record['data']['rows'] == [{'claim_count':5}]
        assert record['data']['sources'][0]['source_rows'] == [2,3,4,5,6]
        return 'There are five claims [E1].', {}
    monkeypatch.setattr(agent, '_call_synthesis_llm', synth)
    assert 'five' in agent.run_plan_tables_agent(**prepare_local_file(insurance_workbook), query='How many claims?')


def test_duplicate_request_reuses_evidence(monkeypatch, insurance_workbook):
    state = {**prepare_local_file(insurance_workbook), 'query':'Read About'}
    state.update(agent.node_init(state))
    state['messages'] = [{'content':[{'text':json.dumps({'tool':'read_sheet','parameters':{'sheet_name':'About'}})}]}]
    first = agent.node_execute_tool(state)
    state['evidence'] = first['evidence']
    dispatch = Mock(side_effect=AssertionError('Must not execute twice'))
    monkeypatch.setattr(agent, '_execute_tool', dispatch)
    second = agent.node_execute_tool(state)
    assert 'evidence' not in second
    assert not dispatch.called


def test_working_context_has_question_and_explicit_omissions():
    records = [{'id':f'E{i}', 'tool':'read_sheet','parameters':{'sheet_name':str(i)},
                'data':{'rows':['x'*400]*20}} for i in range(10)]
    message = planner_messages({'query':'Keep this question','evidence':records})[0]['content'][0]['text']
    assert 'Keep this question' in message
    assert 'omitted_from_working_context' in message
    assert len(message) < 30000
