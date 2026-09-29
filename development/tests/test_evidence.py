from pathlib import Path
import json
from unittest.mock import Mock

import backend.agents.document_agent as agent
from backend.storage.local import prepare_local_file
from backend.preprocessing.documents import document_from_table_inputs




def test_initial_prompt_omits_profiles_and_source_row_arrays(insurance_workbook):
    doc = document_from_table_inputs(**prepare_local_file(insurance_workbook))
    payload = json.dumps(doc.summary())
    assert 'top_values' not in payload
    assert 'source_rows' not in payload
    assert 'Meadow Mapping' in payload


def test_graph_uses_query_and_structured_evidence(insurance_workbook, native_script):
    doc = document_from_table_inputs(**prepare_local_file(insurance_workbook))
    native_script([
        {'tool':'table_tool','parameters':{'document_id':doc.id, 'name':'query_table',
         'parameters':{'sheet_name':'Claim Samples','aggregations':[{'op':'count_rows','as':'claim_count'}]}}},
        {'tool':'answer','parameters':{}}, 'There are five claims [E1].'])
    result = agent.run_document_agent({doc.id:doc}, [doc.id], 'How many claims?')
    record = result['evidence'][0]
    assert record['id'] == 'E1'
    assert record['data']['table_result']['rows'] == [{'claim_count':5}]
    assert record['data']['table_result']['sources'][0]['source_rows'] == [2,3,4,5,6]


def test_duplicate_request_reuses_evidence(monkeypatch,insurance_workbook,native_script):
    doc = document_from_table_inputs(**prepare_local_file(insurance_workbook))
    action = {'tool':'table_tool','parameters':{'document_id':doc.id,'name':'read_sheet',
                                             'parameters':{'sheet_name':'About'}}}
    requests = native_script([action, action, {'tool':'answer','parameters':{}}, 'About [E1].'])
    dispatch = Mock(wraps=agent.execute_document_tool)
    monkeypatch.setattr(agent,'execute_document_tool',dispatch)
    agent.run_document_agent({doc.id:doc}, [doc.id], 'Read About')
    assert dispatch.call_count == 1
    assert 'already_retrieved' in requests[2]['messages'][-1]['content'][0]['toolResult']['content'][0]['text']
