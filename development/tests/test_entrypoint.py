import json

import pytest
from botocore.exceptions import ClientError

from apps.api.lambda_handler import lambda_handler
from backend.agents import document_agent
import inspect

from backend.entrypoint import STATUS_CODES, ask_question, ask_question_local, classify
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
    native_script([{'tool': 'search_documents', 'parameters': {'query': 'Alpha rate', 'document_ids': [doc_id]}},
                   {'tool': 'answer', 'parameters': {}}, 'The member rate is USD 88 [E1].'])


def test_answer_from_saved_document_keys(native_script, store, saved):
    answered(native_script, store.load(saved).id)
    response = ask_question('What is the Alpha rate?', document_keys=[saved], session_id='s1', app='test')
    assert response['status'] == 'answered' and response['status_code'] == 200
    assert response['answer'] == 'The member rate is USD 88 [E1].' and response['message'] is None
    assert response['documents'][0]['name'] == 'rates.txt' and response['documents'][0]['key'] == saved
    [source] = response['sources']
    assert source['id'] == 'E1' and source['passages'][0] == {
        'filename': 'rates.txt', 'location': 'paragraph 1', 'text': 'Plan Alpha member rate is USD 88 per month.'}
    assert response['usage']['llm_calls'] == 3 and response['session_id'] == 's1' and response['error'] is None
    json.dumps(response)  # API callers serialize the response as is


@pytest.mark.parametrize('kwargs, message', [
    ({'question': ' '}, 'question is required'),
    ({'document_keys': []}, 'at least one document'),
    ({'document_keys': ['k'] * 11}, 'at most 10'),
    ({'model': 'unknown'}, 'model must be one of'),
    ({'ambiguity': 'sometimes'}, 'ambiguity must be one of'),
    ({'history': [{'role': 'system', 'content': 'x'}]}, 'history items'),
    ({'document_keys': ['']}, 'nonempty strings'),
])
def test_invalid_requests_are_reported_and_audited(kwargs, message, isolated_audit):
    request = {'question': 'Rate?', 'document_keys': ['k'], **kwargs}
    response = ask_question(request.pop('question'), app='test', **request)
    assert response['status'] == 'invalid_request' and response['status_code'] == 400
    assert message in response['message'] and response['answer'] is None
    lines = (next((isolated_audit / 'metadata' / 'questions').glob('*.jsonl'))).read_text().splitlines()
    assert json.loads(lines[-1])['status'] == 'invalid_request'


def test_missing_saved_document(store):
    response = ask_question('Rate?', document_keys=['document-agent/documents/preprocessed/nope/manifest.json'])
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
                                          ('ExpiredTokenException', 'credentials_expired'),
                                          ('AccessDeniedException', 'error')])
def test_aws_errors_are_classified(monkeypatch, code, status):
    doc = ingest_document('rates.txt', b'USD 88')
    def fail(*a, **kw):
        raise ClientError({'Error': {'Code': code, 'Message': 'no'}}, 'Converse')
    monkeypatch.setattr(document_agent, 'run_document_agent', fail)
    response = ask_question_local('Rate?', documents=[doc])
    assert response['status'] == status and response['status_code'] == STATUS_CODES[status]
    assert response['error']['type'] == 'ClientError'


def test_login_refresh_is_credentials_expired():
    LoginRefreshRequired = type('LoginRefreshRequired', (Exception,), {})
    assert classify(LoginRefreshRequired('renew')) == 'credentials_expired'


def test_lambda_handler_maps_the_event(native_script, store, saved):
    from apps.api import lambda_handler as module
    module.clients.cache_clear()
    answered(native_script, store.load(saved).id)
    response = lambda_handler({'question': 'What is the Alpha rate?', 'session_id': 's9', 'document_keys': [saved],
                               'chat_history': [], 'args': {'model': 'maverick'}}, None)
    assert response['status_code'] == 200 and response['session_id'] == 's9' and 'USD 88' in response['answer']
    assert lambda_handler({'document_keys': [saved]}, None)['status_code'] == 400
    assert module.clients.cache_info().misses == 1  # store and audit sink created once
    module.clients.cache_clear()


def test_production_signature_has_no_development_hooks():
    production = set(inspect.signature(ask_question).parameters)
    assert production == {'question', 'document_keys', 'history', 'session_id', 'clarification', 'model',
                          'ambiguity', 'app', 'store', 'sink'}
    assert {'documents', 'on_step'} <= set(inspect.signature(ask_question_local).parameters)


def test_local_entry_point_validates_the_same_way():
    response = ask_question_local('Rate?', documents=[])
    assert response['status'] == 'invalid_request' and 'at least one document' in response['message']
