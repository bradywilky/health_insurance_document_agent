import json

import pytest

from development.tests.conftest import archived
from botocore.exceptions import ClientError

from apps.api.lambda_handler import lambda_handler
from backend.agents import document_agent
import inspect

from backend.entrypoint import STATUS_CODES, ask_question, ask_question_local
from backend.preprocessing.documents import ingest_document
from backend.storage.s3 import configured_store


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setenv('DOCUMENTS_STORAGE', 'local')
    monkeypatch.setenv('DOCUMENTS_LOCAL_DIR', str(tmp_path / 'library'))
    return configured_store()


@pytest.fixture
def saved(store):
    doc = ingest_document('rates.txt', b'Plan Alpha member rate is USD 88 per month.', store=store)
    return doc.storage_ref['manifest_key']


def answered(native_script, doc_id):
    native_script([{'tool': 'search', 'parameters': {'query': 'Alpha rate'}},
                   {'tool': 'answer', 'parameters': {}}, 'The member rate is USD 88 [E1].'])


def test_answer_from_saved_filenames(native_script, store, saved):
    answered(native_script, store.load(saved).id)
    response = ask_question('What is the Alpha rate?', document_filenames=['rates.txt'], session_id='s1', app_id='test')
    assert response['status'] == 'answered' and response['status_code'] == 200
    assert response['answer'] == 'The member rate is USD 88.' and response['message'] is None
    assert response['documents'][0]['name'] == 'rates.txt' and response['documents'][0]['key'] == saved
    [source] = response['sources']
    assert source['id'] == 'E1' and source['passages'][0] == {
        'filename': 'rates.txt', 'location': 'paragraph 1', 'text': 'Plan Alpha member rate is USD 88 per month.'}
    assert response['usage']['llm_calls'] == 3 and response['session_id'] == 's1' and response['error'] is None
    json.dumps(response)  # API callers serialize the response as is


@pytest.mark.parametrize('kwargs, message', [
    ({'question': ' '}, 'question is required'),
    ({'document_filenames': []}, 'at least one document'),
    ({'document_filenames': ['k'] * 11}, 'at most 10'),
    ({'model': 'unknown'}, 'model must be one of'),
    ({'ambiguity': 'sometimes'}, 'ambiguity must be one of'),
    ({'history': [{'role': 'system', 'content': 'x'}]}, 'history items'),
    ({'document_filenames': ['']}, 'nonempty strings'),
])
def test_invalid_requests_are_reported_and_archived(kwargs, message, isolated_archive):
    request = {'question': 'Rate?', 'document_filenames': ['k'], **kwargs}
    response = ask_question(request.pop('question'), app_id='test', **request)
    assert response['status'] == 'invalid_request' and response['status_code'] == 400
    assert message in response['message'] and response['answer'] is None
    [trace] = archived(isolated_archive, 'document_agent')
    root = trace['spans'][0]
    assert root['meta']['metadata']['status'] == 'invalid_request' and trace['status'] == 'error'
    assert root['meta']['metadata']['request_id'] == response['request_id']


@pytest.mark.parametrize('key', ['rates.txt', 'documents/preprocessed/rates.txt/manifest.json'])
def test_documents_load_by_filename_or_manifest_key(store, saved, key):
    assert store.load(key).storage_ref['manifest_key'] == saved


@pytest.mark.parametrize('key', ['../rates.txt', 'sub/rates.txt'])
def test_filename_keys_cannot_escape(store, saved, key):
    response = ask_question('Rate?', document_filenames=[key])
    assert response['status'] == 'documents_unavailable'


def test_group_name_selects_the_group_folder(native_script, store):
    doc = ingest_document('rates.txt', b'Plan Alpha member rate is USD 88 per month.', store=store.for_group('aol.com'))
    assert doc.storage_ref['manifest_key'] == 'aol.com/documents/preprocessed/rates.txt/manifest.json'
    answered(native_script, doc.id)
    response = ask_question('What is the Alpha rate?', group_name='aol.com', document_filenames=['rates.txt'])
    assert response['status'] == 'answered'
    assert ask_question('Rate?', group_name='other.com', document_filenames=['rates.txt'])['status']         == 'documents_unavailable'


@pytest.mark.parametrize('group', ['', 'a/b', '..'])
def test_invalid_group_name(store, group):
    response = ask_question('Rate?', group_name=group, document_filenames=['rates.txt'])
    assert response['status'] == 'invalid_request' and 'group_name' in response['message']


def test_s3_requires_a_group_name(monkeypatch):
    from development.tests.test_s3_documents import MemoryS3
    from backend.storage.s3 import S3DocumentStore
    monkeypatch.setattr('backend.entrypoint.configured_store', lambda **kwargs: S3DocumentStore(MemoryS3(), 'b'))
    response = ask_question('Rate?', document_filenames=['rates.txt'])
    assert response['status'] == 'invalid_request' and 'group_name is required' in response['message']


def test_missing_saved_document(store):
    response = ask_question('Rate?', document_filenames=['document-agent/documents/preprocessed/nope/manifest.json'])
    assert response['status'] == 'documents_unavailable' and response['status_code'] == 404


def test_no_evidence_is_its_own_status(monkeypatch):
    doc = ingest_document('rates.txt', b'USD 88')
    monkeypatch.setattr(document_agent, 'run_document_agent', lambda *a, **kw: {
        'status': 'no_evidence', 'answer': 'I could not retrieve supporting content.', 'evidence': [],
        'limitations': []})
    response = ask_question_local('Rate?', documents=[doc])
    assert response['status'] == 'no_evidence' and response['status_code'] == 204
    assert response['answer'] is None and 'could not retrieve' in response['message']


def test_clarification_carries_the_original_question(monkeypatch):
    doc = ingest_document('rates.txt', b'USD 88')
    monkeypatch.setattr(document_agent, 'run_document_agent', lambda *a, **kw: {
        'status': 'needs_clarification', 'answer': 'Which rate do you mean?',
        'clarification': {'question': 'Which rate do you mean?', 'options': ['member', 'provider'], 'why': ''}})
    response = ask_question_local('Rate?', documents=[doc], ambiguity='ask')
    assert response['status'] == 'needs_clarification' and response['answer'] == 'Which rate do you mean?'
    assert response['clarification']['original_question'] == 'Rate?'
    assert response['clarification']['options'] == ['member', 'provider']


@pytest.mark.parametrize('code, status', [('ThrottlingException', 'model_unavailable'),
                                          ('AccessDeniedException', 'error')])
def test_aws_errors_are_classified(monkeypatch, code, status):
    doc = ingest_document('rates.txt', b'USD 88')
    def fail(*a, **kw):
        raise ClientError({'Error': {'Code': code, 'Message': 'no'}}, 'Converse')
    monkeypatch.setattr(document_agent, 'run_document_agent', fail)
    response = ask_question_local('Rate?', documents=[doc])
    assert response['status'] == status and response['status_code'] == STATUS_CODES[status]
    assert response['error']['type'] == 'ClientError'


def test_lambda_handler_maps_the_event(native_script, store, saved):
    answered(native_script, store.load(saved).id)
    response = lambda_handler({'question': 'What is the Alpha rate?', 'session_id': 's9', 'user_id': 'u-1042',
                               'document_filenames': [saved],
                               'chat_history': [], 'args': {'model': 'maverick'}}, None)
    assert response['status_code'] == 200 and response['session_id'] == 's9' and 'USD 88' in response['answer']
    assert lambda_handler({'document_filenames': [saved]}, None)['status_code'] == 400
    assert configured_store(read_only=True) is configured_store(read_only=True)  # reused across requests


def test_production_signature_has_no_development_hooks():
    production = set(inspect.signature(ask_question).parameters)
    assert production == {'question', 'document_filenames', 'group_name', 'history', 'session_id', 'user_id',
                          'clarification', 'model', 'ambiguity', 'app_id', 'retrieval_option'}
    assert {'documents', 'on_step'} <= set(inspect.signature(ask_question_local).parameters)


def test_local_entry_point_validates_the_same_way():
    response = ask_question_local('Rate?', documents=[])
    assert response['status'] == 'invalid_request' and 'at least one document' in response['message']


def test_retrieval_option_1_answers_from_the_whole_file_in_one_call(native_script, store, monkeypatch):
    import io
    import pandas as pd
    out = io.BytesIO()
    pd.DataFrame({'Plan': ['Alpha', 'Beta'], 'Rate': [88, 90]}).to_excel(out, sheet_name='Rates', index=False)
    ingest_document('rates.xlsx', out.getvalue(), store=store)
    requests = native_script(['Plan Beta costs 90 (sheet Rates).'])
    response = ask_question('What does Beta cost?', document_filenames=['rates.xlsx'], retrieval_option=1)
    assert response['status'] == 'answered' and response['answer'] == 'Plan Beta costs 90 (sheet Rates).'
    [request] = requests  # one model call, no tools
    assert 'toolConfig' not in request
    prompt = request['messages'][0]['content'][0]['text']
    assert '# Document: rates.xlsx' in prompt and 'Plan,Rate\nAlpha,88\nBeta,90' in prompt
    assert prompt.endswith('Based on the above, What does Beta cost?')
    assert response['usage']['llm_calls'] == 1 and response['protocol']['retrieval_option'] == 1
    [source] = response['sources']
    assert source['tool'] == 'full_context' and source['passages'] == []

    from backend.agents import full_context
    monkeypatch.setattr(full_context, 'MAX_CONTEXT_CHARS', 10)
    response = ask_question('What does Beta cost?', document_filenames=['rates.xlsx'], retrieval_option=1)
    assert response['status'] == 'invalid_request' and 'Use retrieval option 0' in response['message']


@pytest.mark.parametrize('kwargs, message', [
    ({'retrieval_option': 7}, 'retrieval_option must be one of'),
    ({'retrieval_option': '1'}, 'retrieval_option must be one of'),
    ({'retrieval_option': 1, 'ambiguity': 'ask'}, "use ambiguity 'off'"),
])
def test_invalid_retrieval_options_are_rejected(kwargs, message, isolated_archive):
    response = ask_question('Rate?', document_filenames=['k'], app_id='test', **kwargs)
    assert response['status'] == 'invalid_request' and message in response['message']


def test_retrieval_option_1_reads_word_paragraphs_and_tables(native_script, store):
    import io
    from docx import Document as WordDocument
    word = WordDocument()
    word.add_paragraph('Plan Alpha covers emergency care.')
    table = word.add_table(rows=2, cols=2)
    for cell, text in zip(table._cells, ['Plan', 'Copay', 'Alpha', 'USD 150']):
        cell.text = text
    out = io.BytesIO()
    word.save(out)
    ingest_document('benefits.docx', out.getvalue(), store=store)
    requests = native_script(['The Alpha copay is USD 150.'])
    response = ask_question('What is the Alpha copay?', document_filenames=['benefits.docx'], retrieval_option=1)
    assert response['status'] == 'answered' and response['answer'] == 'The Alpha copay is USD 150.'
    prompt = requests[0]['messages'][0]['content'][0]['text']
    assert prompt.startswith('# Document: benefits.docx\nPlan Alpha covers emergency care.')
    assert 'Plan | Copay' in prompt and 'Alpha | USD 150' in prompt
