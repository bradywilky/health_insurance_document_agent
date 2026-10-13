from pathlib import Path
import json
from unittest.mock import Mock

import backend.agents.document_agent as agent

from backend.preprocessing.documents import ingest_document




def test_initial_prompt_omits_profiles_and_source_row_arrays(insurance_workbook):
    doc = ingest_document(insurance_workbook.name, insurance_workbook.read_bytes())
    payload = json.dumps(doc.summary())
    assert 'top_values' not in payload
    assert 'source_rows' not in payload
    assert 'Meadow Mapping' in payload


def test_graph_uses_query_and_structured_evidence(insurance_workbook, native_script):
    doc = ingest_document(insurance_workbook.name, insurance_workbook.read_bytes())
    native_script([
        {'tool':'table_tool','parameters':{'document_id':doc.id, 'name':'query_table',
         'parameters':{'sheet_name':'Claim Samples','aggregations':[{'op':'count_rows','as':'claim_count'}]}}},
        {'tool':'answer','parameters':{}}, 'There are five claims [E1].'])
    result = agent.run_document_agent({doc.id:doc}, [doc.id], 'How many claims?')
    record = result['evidence'][0]
    assert record['id'] == 'E1'
    assert record['data']['table_result']['rows'] == [{'claim_count':5}]
    assert record['data']['table_result']['sources'][0]['source_rows'] == [2,3,4,5,6]


def test_list_sheets_answers_structure_questions(insurance_workbook, native_script):
    doc = ingest_document(insurance_workbook.name, insurance_workbook.read_bytes())
    requests = native_script([
        {'tool':'table_tool','parameters':{'document_id':doc.id,'name':'list_sheets','parameters':{}}},
        {'tool':'answer','parameters':{}}, 'The workbook has About, Claim Samples and other sheets.'])
    result = agent.run_document_agent({doc.id:doc}, [doc.id], 'What sheets are in this file?')
    assert result['status'] == 'answered' and result['protocol']['validation_errors'] == 0
    assert result['coverage']['examined'] == [doc.name]
    writer = json.loads(requests[-1]['messages'][0]['content'][0]['text'])
    assert 'Claim Samples' in json.dumps(writer['evidence'])


def test_search_all_sheets_counts_only_with_matches(insurance_workbook, native_script):
    doc = ingest_document(insurance_workbook.name, insurance_workbook.read_bytes())
    search = lambda term: {'tool':'table_tool','parameters':{'document_id':doc.id,'name':'search_all_sheets',
                                                             'parameters':{'term':term}}}
    native_script([search('zzz-not-present'), {'tool':'answer','parameters':{}},
                   search('DEMO-REMOTE'), {'tool':'answer','parameters':{}}, 'DEMO-REMOTE is listed.'])
    result = agent.run_document_agent({doc.id:doc}, [doc.id], 'Is DEMO-REMOTE listed?')
    assert result['status'] == 'answered' and result['protocol']['validation_errors'] == 1
    assert 'Service Codes' in result['evidence'][1]['data']['table_result']


def test_duplicate_request_reuses_evidence(monkeypatch,insurance_workbook,native_script):
    doc = ingest_document(insurance_workbook.name, insurance_workbook.read_bytes())
    action = {'tool':'table_tool','parameters':{'document_id':doc.id,'name':'read_sheet',
                                             'parameters':{'sheet_name':'About'}}}
    requests = native_script([action, action, {'tool':'answer','parameters':{}}, 'About [E1].'])
    dispatch = Mock(wraps=agent.execute_document_tool)
    monkeypatch.setattr(agent,'execute_document_tool',dispatch)
    agent.run_document_agent({doc.id:doc}, [doc.id], 'Read About')
    assert dispatch.call_count == 1
    assert 'already_retrieved' in requests[2]['messages'][-1]['content'][0]['toolResult']['content'][0]['text']
