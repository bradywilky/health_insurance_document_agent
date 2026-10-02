"""Application audit records: one per question and one per document event.

Two tiers, written to separate locations so they can have separate access policies and retention:

metadata  always written. Who asked, when, which documents, model, prompt and code versions, status, tools
          used, evidence locations, tokens and timings. Questions and answers appear only as SHA-256 hashes.
content   written only when AUDIT_CONTENT=true. Question, answer and evidence. It can contain PHI; store it
          under a restricted prefix with its own KMS key and retention.

Per-call prompts, responses, tokens and timings are in Datadog LLM Observability (backend/observability/llmobs.py);
each record's trace_id finds its trace there.

AUDIT_STORAGE=local (default)  JSON lines under AUDIT_LOCAL_DIR (default data/audit)
AUDIT_STORAGE=s3               one object per record under AUDIT_S3_BUCKET (default DOCUMENTS_S3_BUCKET)
AUDIT_STORAGE=off              no records
AUDIT_REQUIRED=true            a failed audit write fails the request instead of logging a warning
"""
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import threading
import uuid

SCHEMA_VERSION = 2
logger = logging.getLogger(__name__)
_code_version = None


def _flag(name, default='false'):
    return os.getenv(name, default).strip().lower() == 'true'


def content_enabled():
    return _flag('AUDIT_CONTENT')


def audit_required():
    return _flag('AUDIT_REQUIRED')


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def new_id():
    return uuid.uuid4().hex


def digest(text):
    return sha256((text or '').encode('utf-8')).hexdigest()


def code_version():
    """APP_VERSION when the deployment sets it (e.g. a git SHA); otherwise a hash of the backend source."""
    global _code_version
    if os.getenv('APP_VERSION'):
        return os.getenv('APP_VERSION')
    if _code_version is None:
        root = Path(__file__).resolve().parents[1]
        h = sha256()
        for path in sorted(root.rglob('*.py')):
            h.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes())
        _code_version = 'src-' + h.hexdigest()[:12]
    return _code_version


def project_path(path):
    """Relative paths resolve from the project root, like DOCUMENTS_LOCAL_DIR, not the launch directory."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


class LocalAuditSink:
    """Append-only JSON lines: {directory}/{tier}/{kind}/YYYY-MM-DD.jsonl."""

    def __init__(self, directory):
        self.directory = project_path(directory)
        self._lock = threading.Lock()

    def write(self, tier, kind, record):
        path = self.directory / tier / kind / f"{record['timestamp'][:10]}.jsonl"
        line = json.dumps(record, ensure_ascii=False, default=str) + '\n'
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as out:
                out.write(line)


class S3AuditSink:
    """One object per record: {prefix}audit/{tier}/{kind}/YYYY/MM/DD/{timestamp}-{id}.json.

    Objects are never rewritten, so S3 Object Lock (compliance mode) and lifecycle rules can apply per tier."""

    def __init__(self, client, bucket, prefix='document-agent/', kms_key_id=None):
        self.client, self.bucket, self.kms_key_id = client, bucket, kms_key_id
        self.prefix = prefix if not prefix or prefix.endswith('/') else prefix + '/'

    def key(self, tier, kind, record):
        ts = record['timestamp']
        stamp = ts.replace(':', '').replace('-', '').replace('.', '')
        return (f"{self.prefix}audit/{tier}/{kind}/{ts[:4]}/{ts[5:7]}/{ts[8:10]}/"
                f"{stamp}-{record.get('request_id') or record.get('event_id')}.json")

    def write(self, tier, kind, record):
        extra = {'ContentType': 'application/json'}
        if self.kms_key_id:
            extra.update(ServerSideEncryption='aws:kms', SSEKMSKeyId=self.kms_key_id)
        body = json.dumps(record, ensure_ascii=False, default=str).encode('utf-8')
        self.client.upload_fileobj(BytesIO(body), self.bucket, self.key(tier, kind, record), ExtraArgs=extra)


class NullAuditSink:
    def write(self, tier, kind, record):
        return None


def configured_audit_sink():
    mode = os.getenv('AUDIT_STORAGE', 'local').strip().lower()
    if mode == 'off':
        return NullAuditSink()
    if mode == 'local':
        return LocalAuditSink(os.getenv('AUDIT_LOCAL_DIR', 'data/audit'))
    if mode != 's3':
        raise ValueError('AUDIT_STORAGE must be off, local or s3')
    bucket = (os.getenv('AUDIT_S3_BUCKET') or os.getenv('DOCUMENTS_S3_BUCKET') or '').strip()
    if not bucket:
        raise ValueError('AUDIT_S3_BUCKET or DOCUMENTS_S3_BUCKET is required in s3 audit mode')
    import boto3
    from botocore.config import Config
    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE') or None,
                            region_name=os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1')
    client = session.client('s3', config=Config(connect_timeout=5, read_timeout=30,
                                                retries={'mode': 'standard', 'total_max_attempts': 3}))
    return S3AuditSink(client, bucket, os.getenv('AUDIT_S3_PREFIX', os.getenv('DOCUMENTS_S3_PREFIX', 'document-agent/')),
                       kms_key_id=os.getenv('AUDIT_S3_KMS_KEY_ID') or os.getenv('DOCUMENTS_S3_KMS_KEY_ID') or None)


def write(sink, records):
    """records: [(tier, kind, record)]. A failure raises only when AUDIT_REQUIRED=true."""
    for tier, kind, record in records:
        try:
            sink.write(tier, kind, record)
        except Exception as exc:
            if audit_required():
                raise RuntimeError(f'Audit record could not be written: {type(exc).__name__}: {exc}') from exc
            logger.warning('Audit %s/%s record not written: %s: %s', tier, kind, type(exc).__name__, exc)


# ---- record builders -------------------------------------------------------------------------------------

def document_summary(doc):
    ref = getattr(doc, 'storage_ref', None) or {}
    return {'id': doc.id, 'name': doc.name, 'kind': doc.kind, 'manifest_key': ref.get('manifest_key'),
            'has_profile': bool(getattr(doc, 'profile', None))}


def _locations(value, found, limit=20):
    """Collect location labels (page, sheet/row) from evidence data without copying any text."""
    if len(found) >= limit:
        return found
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'location':
                label = (' '.join(f'{k} {v}' for k, v in item.items() if isinstance(v, (str, int)))
                         if isinstance(item, dict) else str(item))
                if label and label not in found:
                    found.append(label)
            elif key == 'source_rows' and isinstance(item, list):
                found.extend(f'row {r}' for r in item[:limit - len(found)])
            else:
                _locations(item, found, limit)
    elif isinstance(value, list):
        for item in value:
            _locations(item, found, limit)
    return found


def evidence_summary(evidence):
    out = []
    for e in evidence or []:
        params, data = e.get('parameters') or {}, e.get('data') or {}
        table = data.get('table_result') or {}
        out.append({'id': e.get('id'), 'tool': e.get('tool'), 'auto': bool(e.get('auto')),
                    'document_id': params.get('document_id'),
                    'operation': params.get('name') if e.get('tool') == 'table_tool' else None,
                    'error': bool(data.get('error') or table.get('error')),
                    'matched_rows': table.get('matched_rows'),
                    'locations': _locations(data, [])})
    return out


def question_records(*, request_id, timestamp, app, session_id, model, ambiguity, prompt_version,
                     documents, question, clarification, result, usage, duration_ms, error, trace_id):
    """Metadata record (always) and content record (AUDIT_CONTENT=true) for one question."""
    result = result or {}
    metadata = {
        'schema_version': SCHEMA_VERSION, 'record_type': 'question', 'request_id': request_id,
        'timestamp': timestamp, 'session_id': session_id, 'app': app,
        'code_version': code_version(), 'prompt_version': prompt_version, 'model': model,
        'ambiguity_mode': ambiguity, 'clarified': bool(clarification), 'documents': documents,
        'question_sha256': digest(question), 'question_chars': len(question or ''),
        'status': result.get('status') or ('error' if error else None),
        'answer_sha256': digest(result['answer']) if result.get('answer') else None,
        'protocol': result.get('protocol'), 'coverage': result.get('coverage'),
        'limitations_count': len(result.get('limitations') or []),
        'evidence': evidence_summary(result.get('evidence')), 'totals': usage,
        'duration_ms': duration_ms, 'error': error, 'trace_id': trace_id,
        'content_recorded': content_enabled()}
    records = [('metadata', 'questions', metadata)]
    if content_enabled():
        records.append(('content', 'questions', {
            'schema_version': SCHEMA_VERSION, 'record_type': 'question_content', 'request_id': request_id,
            'timestamp': timestamp, 'question': question, 'clarification': clarification,
            'answer': result.get('answer'), 'interpretation': result.get('interpretation'),
            'clarification_request': result.get('clarification'), 'limitations': result.get('limitations'),
            'evidence': result.get('evidence')}))
    return records


def record_document_event(event, *, app, sink=None, **fields):
    """Audit an upload or profile change. Never raises unless AUDIT_REQUIRED=true."""
    from backend.observability import llmobs
    write(sink if sink is not None else configured_audit_sink(),
          document_event(event, app=app, trace_id=llmobs.current_trace_id(), **fields))


def document_event(event, *, app, document=None, filename=None, content=None, details=None, error=None,
                   trace_id=None):
    """Metadata record for document_uploaded, upload_failed, profile_generated or profile_reviewed."""
    record = {'schema_version': SCHEMA_VERSION, 'record_type': 'document_event', 'event': event,
              'event_id': new_id(), 'timestamp': now(), 'app': app,
              'code_version': code_version(), 'filename': filename or (document.name if document else None),
              'document': document_summary(document) if document else None, 'error': error,
              'trace_id': trace_id, **(details or {})}
    if content is not None:
        record.update(content_sha256=sha256(content).hexdigest(), size_bytes=len(content))
    if document is not None:
        record['warnings_count'] = len(getattr(document, 'warnings', []) or [])
    return [('metadata', 'documents', record)]
