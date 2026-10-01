import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from streamlit.testing.v1 import AppTest

from backend.agents import document_agent
from backend.observability import audit, tracing
from backend.preprocessing.documents import ingest_document
from backend.services.questions import answer_question, record_document_event

QUESTION = 'What is the member rate for plan Alpha?'
APPS = Path(__file__).resolve().parents[2] / 'apps'


def records(directory, tier, kind):
    files = sorted((directory / tier / kind).glob('*.jsonl'))
    return [json.loads(line) for f in files for line in f.read_text(encoding='utf-8').splitlines()]


@pytest.fixture
def spans(monkeypatch):
    """Route spans to memory without touching the global tracer provider."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, 'tracer', lambda: provider.get_tracer('test'))
    return exporter


@pytest.fixture
def rates_run(native_script):
    doc = ingest_document('rates.txt', b'Plan Alpha member rate is USD 88 per month.')
    native_script([{'tool': 'search_documents', 'parameters': {'query': 'Alpha rate', 'document_ids': [doc.id]}},
                   {'tool': 'answer', 'parameters': {}}, 'The member rate is USD 88 [E1].'])
    return doc


def test_metadata_record_has_no_question_or_answer_text(rates_run, isolated_audit):
    result = answer_question({rates_run.id: rates_run}, [rates_run.id], QUESTION, app='test',
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
    assert meta['llm_calls'][0]['tool_calls'] == ['search_documents']
    assert meta['totals'] == {'llm_calls': 3, 'input_tokens': 30, 'output_tokens': 15} == result['usage']
    assert not (isolated_audit / 'content').exists()


def test_content_tier_is_opt_in_and_separate(rates_run, isolated_audit, monkeypatch):
    monkeypatch.setenv('AUDIT_CONTENT', 'true')
    answer_question({rates_run.id: rates_run}, [rates_run.id], QUESTION, app='test')
    [meta] = records(isolated_audit, 'metadata', 'questions')
    [content] = records(isolated_audit, 'content', 'questions')
    assert meta['content_recorded'] and content['request_id'] == meta['request_id']
    assert content['question'] == QUESTION and 'USD 88' in content['answer']
    assert len(content['llm_calls']) == 3


def test_failed_question_is_audited_then_raised(monkeypatch, isolated_audit):
    doc = ingest_document('rates.txt', b'USD 88')
    def fail(*args, **kwargs):
        raise RuntimeError('Bedrock unavailable')
    monkeypatch.setattr(document_agent, 'run_document_agent', fail)
    with pytest.raises(RuntimeError, match='unavailable'):
        answer_question({doc.id: doc}, [doc.id], QUESTION, app='test')
    [meta] = records(isolated_audit, 'metadata', 'questions')
    assert meta['status'] == 'error' and 'Bedrock unavailable' in meta['error']


class BrokenSink:
    def write(self, tier, kind, record):
        raise OSError('disk full')


def test_audit_failure_is_a_warning_unless_required(rates_run, monkeypatch, caplog):
    result = answer_question({rates_run.id: rates_run}, [rates_run.id], QUESTION, app='test',
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


def test_spans_follow_the_graph_and_match_the_audit_record(rates_run, spans, isolated_audit):
    result = answer_question({rates_run.id: rates_run}, [rates_run.id], QUESTION, app='test',
                             session_id='s1')
    finished = spans.get_finished_spans()
    by_id = {s.context.span_id: s for s in finished}
    [root] = [s for s in finished if s.name == 'invoke_agent document_agent']
    assert {s.context.trace_id for s in finished} == {root.context.trace_id}
    assert result['trace_id'] == format(root.context.trace_id, '032x')
    assert records(isolated_audit, 'metadata', 'questions')[0]['trace_id'] == result['trace_id']
    assert root.attributes['session.id'] == 's1' and root.attributes['gen_ai.operation.name'] == 'invoke_agent'

    nodes = [s for s in finished if s.name.startswith('graph.node')]
    assert {s.attributes['app.graph.node'] for s in nodes} >= {'init', 'plan', 'tools', 'synthesize'}
    assert all(s.parent.span_id == root.context.span_id for s in nodes)

    chats = [s for s in finished if s.name.startswith('chat ')]
    assert len(chats) == 3
    assert all(by_id[s.parent.span_id].name.startswith('graph.node') for s in chats)
    assert chats[0].attributes['gen_ai.usage.input_tokens'] == 10
    assert chats[0].attributes['app.llm.tool_calls'] == ('search_documents',)
    assert chats[-1].attributes['app.llm.step'] == 'Document Answer'

    [tool] = [s for s in finished if s.name == 'execute_tool search_documents']
    assert by_id[tool.parent.span_id].attributes['app.graph.node'] == 'tools'
    assert tool.attributes['app.document_ids'] == (rates_run.id,) and tool.attributes['app.evidence_id'] == 'E1'
    # No prompt or answer text unless TRACE_CONTENT=true.
    assert not any(s.events for s in finished)
    assert 'Alpha' not in json.dumps([dict(s.attributes) for s in finished])


def test_trace_content_is_opt_in(rates_run, spans, monkeypatch):
    monkeypatch.setenv('TRACE_CONTENT', 'true')
    answer_question({rates_run.id: rates_run}, [rates_run.id], QUESTION, app='test')
    chat = next(s for s in spans.get_finished_spans() if s.name.startswith('chat '))
    assert {e.name for e in chat.events} == {'gen_ai.input', 'gen_ai.output'}


def test_tool_errors_mark_their_span(native_script, spans):
    doc = ingest_document('rates.txt', b'USD 88')
    # 77 is not in any cited source, so the calculation is refused and returned to the planner as an error.
    native_script([{'tool': 'calculate', 'parameters': {'label': 'x', 'expression': '77 * 2', 'sources': ['question']}},
                   {'tool': 'answer', 'parameters': {}}] + ['USD 88 [E1].'] * 4)
    answer_question({doc.id: doc}, [doc.id], 'Rate?', app='test')
    tool = next(s for s in spans.get_finished_spans() if s.name == 'execute_tool calculate')
    assert tool.status.status_code == StatusCode.ERROR


def test_document_events_record_hashes_not_contents(isolated_audit):
    content = b'Plan Alpha member rate is USD 88.'
    doc = ingest_document('rates.txt', content)
    record_document_event('document_uploaded', app='upload', document=doc, content=content,
                          details={'profile_generated': False})
    [event] = records(isolated_audit, 'metadata', 'documents')
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
