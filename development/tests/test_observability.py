import hashlib
import json
from pathlib import Path

import pytest
from ddtrace.llmobs import LLMObs
from streamlit.testing.v1 import AppTest

from backend.agents import document_agent
from backend.observability import archive, tracing
from backend.preprocessing.documents import ingest_document
from backend.entrypoint import ask_question_local
from backend.storage.filesystem import LocalDocumentStore
from development.observability import capture
from development.tests.conftest import archived

QUESTION = 'What is the member rate for plan Alpha?'
APPS = Path(__file__).resolve().parents[2] / 'apps'


@pytest.fixture
def rates_run(native_script):
    doc = ingest_document('rates.txt', b'Plan Alpha member rate is USD 88 per month.')
    native_script([{'tool': 'search_documents', 'parameters': {'query': 'Alpha rate', 'document_ids': [doc.id]}},
                   {'tool': 'answer', 'parameters': {}}, 'The member rate is USD 88 [E1].'])
    return doc


def root_metadata(trace):
    [root] = [e for e in trace['spans'] if e['parent_id'] == 'undefined']
    return root['meta']['metadata']


def test_archived_trace_is_the_captured_trace(rates_run, isolated_archive):
    result = ask_question_local(QUESTION, documents=[rates_run], app_id='test', session_id='s1')
    [trace] = archived(isolated_archive, 'document_agent')
    assert trace['trace_id'] == result['trace_id'] and trace['status'] == 'ok'
    assert trace['spans'] == capture.trace(result['trace_id'])
    assert trace['code_version'] == archive.code_version() and trace['schema_version'] == 1


def test_archived_question_has_audit_fields_but_no_text(rates_run, isolated_archive):
    result = ask_question_local(QUESTION, documents=[rates_run], app_id='test', session_id='s1')
    [trace] = archived(isolated_archive, 'document_agent')
    text = json.dumps(trace)
    assert 'Alpha' not in text and 'USD 88' not in text and 'member rate' not in text
    meta = root_metadata(trace)
    assert meta['question_sha256'] == archive.digest(QUESTION)
    assert meta['answer_sha256'] == archive.digest(result['answer'])
    assert meta['request_id'] == result['request_id'] and trace['spans'][0]['session_id'] == 's1'
    assert meta['status'] == 'answered' and 'error' not in meta
    assert meta['prompt_version'] == document_agent.PROMPT_VERSION
    assert meta['documents'][0]['id'] == rates_run.id and meta['documents'][0]['name'] == 'rates.txt'
    assert [e['tool'] for e in meta['evidence']] == ['search_documents']
    assert meta['evidence'][0]['locations'] == ['paragraph 1']
    assert meta['totals'] == {'llm_calls': 3, 'input_tokens': 30, 'output_tokens': 15} == result['usage']


def test_content_is_archived_only_with_trace_content(rates_run, isolated_archive, monkeypatch):
    monkeypatch.setenv('TRACE_CONTENT', 'true')
    ask_question_local(QUESTION, documents=[rates_run], app_id='test')
    [trace] = archived(isolated_archive, 'document_agent')
    root = trace['spans'][0]
    assert QUESTION in root['meta']['input']['value'] and 'USD 88' in root['meta']['output']['value']
    [tool] = [e for e in trace['spans'] if capture.kind(e) == 'tool']
    assert 'Alpha' in tool['meta']['output']['value']  # the evidence the tool returned


def test_failed_question_is_archived(monkeypatch, isolated_archive):
    doc = ingest_document('rates.txt', b'USD 88')
    def fail(*args, **kwargs):
        raise RuntimeError('Bedrock unavailable')
    monkeypatch.setattr(document_agent, 'run_document_agent', fail)
    response = ask_question_local(QUESTION, documents=[doc], app_id='test')
    assert response['status'] == 'error' and response['status_code'] == 500 and response['answer'] is None
    assert 'Bedrock unavailable' in response['message']
    [trace] = archived(isolated_archive, 'document_agent')
    meta = root_metadata(trace)
    assert trace['status'] == 'error' and meta['status'] == 'error' and 'Bedrock unavailable' in meta['error']
    assert meta['request_id'] == response['request_id']


class BrokenArchive:
    def write(self, trace):
        raise OSError('disk full')


def test_archive_failure_is_a_warning_unless_required(rates_run, monkeypatch, caplog):
    result = ask_question_local(QUESTION, documents=[rates_run], app_id='test', archive=BrokenArchive())
    assert result['status'] == 'answered' and 'disk full' in caplog.text
    monkeypatch.setenv('TRACE_ARCHIVE_REQUIRED', 'true')
    with pytest.raises(RuntimeError, match='Trace could not be archived'):
        ask_question_local(QUESTION, documents=[rates_run], app_id='test', archive=BrokenArchive())


def test_s3_archive_writes_one_encrypted_object_per_trace():
    uploads = []
    class Client:
        def upload_fileobj(self, body, bucket, key, ExtraArgs):
            uploads.append((bucket, key, ExtraArgs, json.loads(body.read())))
    store = archive.S3TraceArchive(Client(), 'audit-bucket', 'agent', kms_key_id='key-1')
    trace = {'trace_id': 'abc', 'name': 'document_agent', 'started_at': '2026-03-04T05:06:07.890+00:00',
             'spans': []}
    store.write(trace)
    [(bucket, key, extra, body)] = uploads
    assert key == 'agent/traces/document_agent/2026/03/04/20260304T050607890+0000-abc.json'
    assert extra == {'ContentType': 'application/json', 'ServerSideEncryption': 'aws:kms', 'SSEKMSKeyId': 'key-1'}
    assert body['trace_id'] == 'abc' and body['spans'] == []


def test_relative_archive_paths_resolve_from_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    root = Path(archive.__file__).resolve().parents[2]
    assert archive.LocalTraceArchive('data/traces').directory == root / 'data' / 'traces'
    assert archive.LocalTraceArchive(tmp_path / 'x').directory == tmp_path / 'x'


def test_archive_modes(monkeypatch):
    monkeypatch.setenv('TRACE_ARCHIVE', 'off')
    assert isinstance(archive.configured_archive(), archive.NullTraceArchive)
    monkeypatch.setenv('TRACE_ARCHIVE', 's3')
    monkeypatch.delenv('TRACE_ARCHIVE_S3_BUCKET', raising=False)
    monkeypatch.delenv('DOCUMENTS_S3_BUCKET', raising=False)
    with pytest.raises(ValueError, match='BUCKET'):
        archive.configured_archive()
    monkeypatch.setenv('TRACE_ARCHIVE', 'database')
    with pytest.raises(ValueError):
        archive.configured_archive()


def test_spans_follow_the_graph(rates_run):
    result = ask_question_local(QUESTION, documents=[rates_run], app_id='test', session_id='s1', user_id='u-1042')
    events = capture.trace(result['trace_id'])
    by_id = {e['span_id']: e for e in events}
    [root] = [e for e in events if capture.kind(e) == 'agent']
    assert root['name'] == 'document_agent' and root['parent_id'] == 'undefined'
    assert root['session_id'] == 's1' and root['status'] == 'ok'
    assert {'app_id:test', 'user_id:u-1042'} <= set(root['tags'])
    assert root['meta']['metadata']['request_id'] == result['request_id']
    assert root['meta']['metadata']['status'] == 'answered'

    nodes = [e for e in events if capture.kind(e) == 'workflow']
    assert {e['name'] for e in nodes} >= {'init', 'plan', 'tools', 'synthesize'}
    assert all(e['parent_id'] == root['span_id'] for e in nodes)

    calls = [e for e in events if capture.kind(e) == 'llm']
    assert len(calls) == 3
    assert all(capture.kind(by_id[e['parent_id']]) == 'workflow' for e in calls)
    assert calls[0]['metrics'] == {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}
    assert calls[0]['meta']['metadata']['tool_calls'] == ['search_documents']
    assert calls[0]['meta']['model_provider'] == 'amazon_bedrock'
    assert calls[-1]['name'] == 'Document Answer'

    [tool] = [e for e in events if capture.kind(e) == 'tool']
    assert tool['name'] == 'search_documents' and by_id[tool['parent_id']]['name'] == 'tools'
    assert tool['meta']['metadata']['document_ids'] == [rates_run.id]
    assert tool['meta']['metadata']['evidence_id'] == 'E1'
    # No prompt, question, answer or tool parameter text unless TRACE_CONTENT=true.
    assert not any(e['meta'].get('input') or e['meta'].get('output') for e in events)
    assert 'Alpha' not in json.dumps(events)


def test_trace_content_is_opt_in(rates_run, monkeypatch):
    monkeypatch.setenv('TRACE_CONTENT', 'true')
    result = ask_question_local(QUESTION, documents=[rates_run], app_id='test')
    events = capture.trace(result['trace_id'])
    root = next(e for e in events if capture.kind(e) == 'agent')
    assert QUESTION in root['meta']['input']['value'] and 'USD 88' in root['meta']['output']['value']
    call = capture.summarize(events)['calls'][0]
    assert [m['role'] for m in call['input']] == ['system', 'user'] and 'Alpha' in call['input'][1]['content']
    assert call['output'][0]['tool_calls'][0]['name'] == 'search_documents'


def test_summary_reports_totals_largest_call_and_gaps(rates_run):
    summary = capture.summarize(capture.trace(ask_question_local(QUESTION, documents=[rates_run])['trace_id']))
    totals, calls = summary['totals'], summary['calls']
    assert {k: totals[k] for k in ('llm_calls', 'input_tokens', 'output_tokens', 'total_tokens')} == {
        'llm_calls': 3, 'input_tokens': 30, 'output_tokens': 15, 'total_tokens': 45}
    assert totals['largest_call'] == 1 and totals['models']
    assert [c['call'] for c in calls] == [1, 2, 3] and calls[0]['gap_ms'] is None
    assert all(c['gap_ms'] >= 0 for c in calls[1:])
    assert [c['start_ms'] for c in calls] == sorted(c['start_ms'] for c in calls)
    assert [t['tool'] for t in summary['tools']] == ['search_documents']


def test_traces_are_recorded_and_archived_without_datadog(rates_run, isolated_archive, monkeypatch):
    monkeypatch.setattr(LLMObs, 'enabled', False)
    result = ask_question_local(QUESTION, documents=[rates_run], app_id='test')
    assert result['status'] == 'answered' and result['usage']['llm_calls'] == 3
    [trace] = archived(isolated_archive, 'document_agent')
    assert trace['trace_id'] == result['trace_id']
    assert len([e for e in trace['spans'] if capture.kind(e) == 'llm']) == 3


def test_tool_errors_mark_their_span(native_script):
    doc = ingest_document('rates.txt', b'USD 88')
    # 77 is not in any cited source, so the calculation is refused and returned to the planner as an error.
    native_script([{'tool': 'calculate', 'parameters': {'label': 'x', 'expression': '77 * 2', 'sources': ['question']}},
                   {'tool': 'answer', 'parameters': {}}] + ['USD 88 [E1].'] * 4)
    result = ask_question_local('Rate?', documents=[doc], app_id='test')
    tool = next(e for e in capture.trace(result['trace_id']) if e['name'] == 'calculate')
    assert tool['status'] == 'error'


def test_document_changes_are_archived_with_hashes_not_contents(isolated_archive, tmp_path):
    content = b'Plan Alpha member rate is USD 88.'
    store = LocalDocumentStore(tmp_path / 'library', group='example.org')
    doc = ingest_document('rates.txt', content, store=store)
    store.save_profile(doc.storage_ref['manifest_key'], {'glossary': [
        {'term': 'PMPM', 'meaning': 'per member per month', 'source': 'confirmed'}], 'meta': {'reviewed': True}})
    store.delete('rates.txt')
    with pytest.raises(ValueError):
        ingest_document('rates.exe', content, store=store)

    first, failed = archived(isolated_archive, 'ingest_document')
    meta = root_metadata(first)
    assert meta['filename'] == 'rates.txt' and meta['group_name'] == 'example.org' and meta['saved']
    assert meta['bytes'] == len(content) and meta['content_sha256'] == hashlib.sha256(content).hexdigest()
    assert meta['document']['id'] == doc.id
    assert failed['status'] == 'error' and 'Unsupported file type' in failed['spans'][0]['meta']['error.message']
    [profile] = archived(isolated_archive, 'save_profile')
    assert root_metadata(profile)['terms_added'] == ['PMPM'] and root_metadata(profile)['terms_confirmed'] == 1
    [deleted] = archived(isolated_archive, 'delete_document')
    assert root_metadata(deleted) == {'filename': 'rates.txt', 'group_name': 'example.org'}
    text = json.dumps([first, profile, deleted])
    assert 'Alpha' not in text and 'per member per month' not in text


def test_versions_are_stable_identifiers(monkeypatch):
    assert document_agent.PROMPT_VERSION.startswith('p-') and len(document_agent.PROMPT_VERSION) == 14
    assert archive.code_version().startswith('src-')
    monkeypatch.setenv('APP_VERSION', 'abc123')
    assert archive.code_version() == 'abc123'


def test_chat_and_upload_apps_archive_their_traces(monkeypatch, tmp_path, isolated_archive):
    monkeypatch.setenv('DOCUMENTS_STORAGE', 'local')
    monkeypatch.setenv('DOCUMENTS_LOCAL_DIR', str(tmp_path / 'library'))
    portal = AppTest.from_file(str(APPS / 'upload' / 'app.py')).run(timeout=20)
    next(b for b in portal.button if b.label == 'Load Bingle-Dingle examples').click().run(timeout=30)
    assert not portal.exception
    uploads = archived(isolated_archive, 'ingest_document')
    assert len(uploads) == 8 and {t['status'] for t in uploads} == {'ok'}

    monkeypatch.setattr(document_agent, 'run_document_agent',
                        lambda *a, **kw: {'status': 'answered', 'answer': 'USD 88 [E1].', 'evidence': [],
                                          'limitations': []})
    chat = AppTest.from_file(str(APPS / 'chat' / 'app.py')).run(timeout=20)
    chat.multiselect[0].set_value([chat.multiselect[0].options[0]]).run()
    chat.chat_input[0].set_value('What is the rate?').run(timeout=20)
    assert not chat.exception
    [question] = archived(isolated_archive, 'document_agent')
    root = question['spans'][0]
    assert 'app_id:chat' in root['tags'] and root['session_id'] == chat.session_state['session_id']
    assert any(root['meta']['metadata']['request_id'][:12] in c.value for c in chat.caption)


class FakeDatadog:
    """Stands in for ddtrace's LLMObs: records what the tracing helpers forward."""
    enabled = True

    def __init__(self):
        self.spans, self.annotations = [], []

    def _start(self, kind):
        from contextlib import contextmanager
        @contextmanager
        def start(name, **options):
            current = type('DatadogSpan', (), {'error': 0, 'set_tag': lambda self, k, v: None})()
            current.ids = {'span_id': f'dd{len(self.spans)}', 'trace_id': 'dd-trace'}
            self.spans.append((kind, name, options))
            yield current
        return start

    def __getattr__(self, kind):
        return self._start(kind)

    def annotate(self, span, **fields):
        self.annotations.append((span.ids['span_id'], fields))

    def export_span(self, span):
        return span.ids


def test_datadog_receives_the_same_spans_and_trace_id(rates_run, monkeypatch, isolated_archive):
    datadog = FakeDatadog()
    monkeypatch.setattr(tracing, 'LLMObs', datadog)
    result = ask_question_local(QUESTION, documents=[rates_run], app_id='test', session_id='s1', user_id='u-1042')
    [trace] = archived(isolated_archive, 'document_agent')
    assert result['trace_id'] == trace['trace_id'] == 'dd-trace'
    assert {'app_id': 'test', 'user_id': 'u-1042'}.items() <= next(
        f['tags'] for span_id, f in datadog.annotations if span_id == trace['spans'][0]['span_id'] and 'tags' in f).items()
    assert [(kind, name) for kind, name, _ in datadog.spans] == [
        (capture.kind(e), e['name']) for e in sorted(trace['spans'], key=lambda e: int(e['span_id'][2:]))]
    assert datadog.spans[0][2] == {'session_id': 's1'}
    for event in trace['spans']:
        sent = {'metadata': {}, 'metrics': {}, 'tags': {}}
        for span_id, fields in datadog.annotations:
            if span_id == event['span_id']:
                for key in sent:
                    sent[key].update(fields.get(key, {}))
        assert sent['metadata'] == event['meta']['metadata'] and sent['metrics'] == event['metrics']
        assert [f'{k}:{v}' for k, v in sent['tags'].items()] == event['tags']
