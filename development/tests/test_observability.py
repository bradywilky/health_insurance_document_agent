import json
from pathlib import Path

import pytest
from ddtrace.llmobs import LLMObs
from streamlit.testing.v1 import AppTest

from backend.agents import document_agent
from backend.observability import audit, llmobs
from backend.preprocessing.documents import ingest_document
from backend.entrypoint import ask_question_local
from backend.observability.audit import record_document_event
from development.observability import capture

QUESTION = 'What is the member rate for plan Alpha?'
APPS = Path(__file__).resolve().parents[2] / 'apps'


def records(directory, tier, kind):
    files = sorted((directory / tier / kind).glob('*.jsonl'))
    return [json.loads(line) for f in files for line in f.read_text(encoding='utf-8').splitlines()]


@pytest.fixture
def rates_run(native_script):
    doc = ingest_document('rates.txt', b'Plan Alpha member rate is USD 88 per month.')
    native_script([{'tool': 'search_documents', 'parameters': {'query': 'Alpha rate', 'document_ids': [doc.id]}},
                   {'tool': 'answer', 'parameters': {}}, 'The member rate is USD 88 [E1].'])
    return doc


def test_metadata_record_has_no_question_or_answer_text(rates_run, isolated_audit):
    result = ask_question_local(QUESTION, documents=[rates_run], app='test',
                             session_id='s1')
    [meta] = records(isolated_audit, 'metadata', 'questions')
    text = json.dumps(meta)
    assert 'Alpha' not in text and 'USD 88' not in text and 'member rate' not in text
    assert meta['question_sha256'] == audit.digest(QUESTION)
    assert meta['answer_sha256'] == audit.digest(result['answer'])
    assert meta['request_id'] == result['request_id'] and meta['session_id'] == 's1'
    assert meta['status'] == 'answered' and meta['error'] is None
    assert meta['prompt_version'] == document_agent.PROMPT_VERSION
    assert meta['documents'][0]['id'] == rates_run.id and meta['documents'][0]['name'] == 'rates.txt'
    assert [e['tool'] for e in meta['evidence']] == ['search_documents']
    assert meta['evidence'][0]['locations'] == ['paragraph 1']
    assert meta['totals'] == {'llm_calls': 3, 'input_tokens': 30, 'output_tokens': 15} == result['usage']
    assert not (isolated_audit / 'content').exists()


def test_content_tier_is_opt_in_and_separate(rates_run, isolated_audit, monkeypatch):
    monkeypatch.setenv('AUDIT_CONTENT', 'true')
    ask_question_local(QUESTION, documents=[rates_run], app='test')
    [meta] = records(isolated_audit, 'metadata', 'questions')
    [content] = records(isolated_audit, 'content', 'questions')
    assert meta['content_recorded'] and content['request_id'] == meta['request_id']
    assert content['question'] == QUESTION and 'USD 88' in content['answer']
    assert 'llm_calls' not in content  # per-call prompts are in Datadog LLM Observability


def test_failed_question_is_audited_then_raised(monkeypatch, isolated_audit):
    doc = ingest_document('rates.txt', b'USD 88')
    def fail(*args, **kwargs):
        raise RuntimeError('Bedrock unavailable')
    monkeypatch.setattr(document_agent, 'run_document_agent', fail)
    response = ask_question_local(QUESTION, documents=[doc], app='test')
    assert response['status'] == 'error' and response['status_code'] == 500 and response['answer'] is None
    assert 'Bedrock unavailable' in response['message']
    [meta] = records(isolated_audit, 'metadata', 'questions')
    assert meta['status'] == 'error' and 'Bedrock unavailable' in meta['error']
    assert meta['request_id'] == response['request_id']


class BrokenSink:
    def write(self, tier, kind, record):
        raise OSError('disk full')


def test_audit_failure_is_a_warning_unless_required(rates_run, monkeypatch, caplog):
    result = ask_question_local(QUESTION, documents=[rates_run], app='test',
                             sink=BrokenSink())
    assert result['status'] == 'answered' and 'disk full' in caplog.text
    monkeypatch.setenv('AUDIT_REQUIRED', 'true')
    with pytest.raises(RuntimeError, match='Audit record could not be written'):
        audit.write(BrokenSink(), [('metadata', 'questions', {'timestamp': audit.now()})])


def test_s3_sink_writes_one_encrypted_object_per_record():
    uploads = []
    class Client:
        def upload_fileobj(self, body, bucket, key, ExtraArgs):
            uploads.append((bucket, key, ExtraArgs, json.loads(body.read())))
    sink = audit.S3AuditSink(Client(), 'audit-bucket', 'agent', kms_key_id='key-1')
    record = {'timestamp': '2026-03-04T05:06:07.890+00:00', 'request_id': 'abc'}
    sink.write('metadata', 'questions', record)
    [(bucket, key, extra, body)] = uploads
    assert key == 'agent/audit/metadata/questions/2026/03/04/20260304T050607890+0000-abc.json'
    assert extra == {'ContentType': 'application/json', 'ServerSideEncryption': 'aws:kms', 'SSEKMSKeyId': 'key-1'}
    assert body == record


def test_relative_audit_paths_resolve_from_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    root = Path(audit.__file__).resolve().parents[2]
    assert audit.LocalAuditSink('data/audit').directory == root / 'data' / 'audit'
    assert audit.LocalAuditSink(tmp_path / 'x').directory == tmp_path / 'x'


def test_audit_storage_modes(monkeypatch):
    monkeypatch.setenv('AUDIT_STORAGE', 'off')
    assert isinstance(audit.configured_audit_sink(), audit.NullAuditSink)
    monkeypatch.setenv('AUDIT_STORAGE', 's3')
    monkeypatch.delenv('AUDIT_S3_BUCKET', raising=False)
    monkeypatch.delenv('DOCUMENTS_S3_BUCKET', raising=False)
    with pytest.raises(ValueError, match='BUCKET'):
        audit.configured_audit_sink()
    monkeypatch.setenv('AUDIT_STORAGE', 'database')
    with pytest.raises(ValueError):
        audit.configured_audit_sink()


def test_spans_follow_the_graph_and_match_the_audit_record(rates_run, isolated_audit):
    result = ask_question_local(QUESTION, documents=[rates_run], app='test', session_id='s1')
    events = capture.trace(result['trace_id'])
    by_id = {e['span_id']: e for e in events}
    [root] = [e for e in events if capture.kind(e) == 'agent']
    assert root['name'] == 'document_agent' and root['parent_id'] == 'undefined'
    assert records(isolated_audit, 'metadata', 'questions')[0]['trace_id'] == result['trace_id']
    assert root['session_id'] == 's1' and 'app:test' in root['tags'] and root['status'] == 'ok'
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
    result = ask_question_local(QUESTION, documents=[rates_run], app='test')
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


def test_runs_unchanged_without_datadog(rates_run, monkeypatch):
    monkeypatch.setattr(LLMObs, 'enabled', False)
    with capture.recording() as events:
        result = ask_question_local(QUESTION, documents=[rates_run], app='test')
    assert result['status'] == 'answered' and result['trace_id'] is None and result['usage']['llm_calls'] == 3
    assert events == []


def test_tool_errors_mark_their_span(native_script):
    doc = ingest_document('rates.txt', b'USD 88')
    # 77 is not in any cited source, so the calculation is refused and returned to the planner as an error.
    native_script([{'tool': 'calculate', 'parameters': {'label': 'x', 'expression': '77 * 2', 'sources': ['question']}},
                   {'tool': 'answer', 'parameters': {}}] + ['USD 88 [E1].'] * 4)
    result = ask_question_local('Rate?', documents=[doc], app='test')
    tool = next(e for e in capture.trace(result['trace_id']) if e['name'] == 'calculate')
    assert tool['status'] == 'error'


def test_document_events_record_hashes_not_contents(isolated_audit):
    content = b'Plan Alpha member rate is USD 88.'
    doc = ingest_document('rates.txt', content)
    with llmobs.span('workflow', 'ingest_document') as current:
        record_document_event('document_uploaded', app='upload', document=doc, content=content,
                              details={'profile_generated': False})
        trace_id = llmobs.trace_id_of(current)
    [event] = records(isolated_audit, 'metadata', 'documents')
    assert event['trace_id'] == trace_id is not None
    assert event['event'] == 'document_uploaded' and event['filename'] == 'rates.txt'
    assert event['size_bytes'] == len(content) and len(event['content_sha256']) == 64
    assert 'Alpha' not in json.dumps(event)


def test_versions_are_stable_identifiers(monkeypatch):
    assert document_agent.PROMPT_VERSION.startswith('p-') and len(document_agent.PROMPT_VERSION) == 14
    assert audit.code_version().startswith('src-')
    monkeypatch.setenv('APP_VERSION', 'abc123')
    assert audit.code_version() == 'abc123'


def test_chat_and_upload_apps_write_audit_records(monkeypatch, tmp_path, isolated_audit):
    monkeypatch.setenv('DOCUMENTS_STORAGE', 'local')
    monkeypatch.setenv('DOCUMENTS_LOCAL_DIR', str(tmp_path / 'library'))
    portal = AppTest.from_file(str(APPS / 'upload' / 'app.py')).run(timeout=20)
    next(b for b in portal.button if b.label == 'Load Bingle-Dingle examples').click().run(timeout=30)
    assert not portal.exception
    uploads = records(isolated_audit, 'metadata', 'documents')
    assert len(uploads) == 8 and {e['event'] for e in uploads} == {'document_uploaded'}

    monkeypatch.setattr(document_agent, 'run_document_agent',
                        lambda *a, **kw: {'status': 'answered', 'answer': 'USD 88 [E1].', 'evidence': [],
                                          'limitations': []})
    chat = AppTest.from_file(str(APPS / 'chat' / 'app.py')).run(timeout=20)
    chat.multiselect[0].set_value([chat.multiselect[0].options[0]]).run()
    chat.chat_input[0].set_value('What is the rate?').run(timeout=20)
    assert not chat.exception
    [meta] = records(isolated_audit, 'metadata', 'questions')
    assert meta['app'] == 'chat' and meta['session_id'] == chat.session_state['session_id']
    assert any(meta['request_id'][:12] in c.value for c in chat.caption)
