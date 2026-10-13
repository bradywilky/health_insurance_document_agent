"""Trace archive: every question and document change, saved where it can be read without Datadog.

Each finished trace from backend/observability/tracing.py (the same spans and fields Datadog LLM Observability
receives) is written as one JSON object. Question and answer text, prompts and tool results appear only with
TRACE_CONTENT=true, exactly as in Datadog; store the archive under a restricted prefix with its own KMS key and
retention if you turn that on.

TRACE_ARCHIVE=local (default)  one file per trace under TRACE_ARCHIVE_LOCAL_DIR (default data/traces/)
TRACE_ARCHIVE=s3               one object per trace in TRACE_ARCHIVE_S3_BUCKET (default DOCUMENTS_S3_BUCKET)
TRACE_ARCHIVE=off              nothing is archived (traces still go to Datadog when it is enabled)
TRACE_ARCHIVE_REQUIRED=true    a failed write fails the request instead of logging a warning

Files are named {root span}/YYYY/MM/DD/{timestamp}-{trace_id}.json; S3 objects add {prefix}traces/ in front. They
are never rewritten, so S3 Object Lock and lifecycle rules can apply.
"""
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import logging
import os
from pathlib import Path

SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)
_code_version = None


def required():
    return os.getenv('TRACE_ARCHIVE_REQUIRED', 'false').strip().lower() == 'true'


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


def relative_name(trace):
    """{root span}/YYYY/MM/DD/{timestamp}-{trace_id}.json"""
    ts = trace['started_at']
    stamp = ts.replace(':', '').replace('-', '').replace('.', '')
    return f"{trace['name']}/{ts[:4]}/{ts[5:7]}/{ts[8:10]}/{stamp}-{trace['trace_id']}.json"


def _body(trace):
    return json.dumps({'schema_version': SCHEMA_VERSION, 'code_version': code_version(), **trace},
                      ensure_ascii=False, default=str).encode('utf-8')


class LocalTraceArchive:
    def __init__(self, directory):
        self.directory = project_path(directory)

    def write(self, trace):
        path = self.directory / relative_name(trace)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_body(trace))


class S3TraceArchive:
    def __init__(self, client, bucket, prefix='', kms_key_id=None):
        self.client, self.bucket, self.kms_key_id = client, bucket, kms_key_id
        self.prefix = prefix if not prefix or prefix.endswith('/') else prefix + '/'

    def write(self, trace):
        extra = {'ContentType': 'application/json'}
        if self.kms_key_id:
            extra.update(ServerSideEncryption='aws:kms', SSEKMSKeyId=self.kms_key_id)
        self.client.upload_fileobj(BytesIO(_body(trace)), self.bucket, self.prefix + 'traces/' + relative_name(trace),
                                   ExtraArgs=extra)


class NullTraceArchive:
    def write(self, trace):
        return None


def configured_archive():
    """The archive the environment selects. Reused while the settings stay the same (e.g. a warm Lambda)."""
    names = ('TRACE_ARCHIVE', 'TRACE_ARCHIVE_LOCAL_DIR', 'TRACE_ARCHIVE_S3_BUCKET', 'TRACE_ARCHIVE_S3_PREFIX',
             'TRACE_ARCHIVE_S3_KMS_KEY_ID', 'DOCUMENTS_S3_BUCKET', 'DOCUMENTS_S3_PREFIX_BASE',
             'DOCUMENTS_S3_KMS_KEY_ID', 'AWS_PROFILE', 'AWS_REGION', 'AWS_DEFAULT_REGION')
    return _archive(tuple((name, os.getenv(name)) for name in names))


@lru_cache(maxsize=4)
def _archive(settings):
    env = {k: (v or '').strip() for k, v in settings}
    mode = (env['TRACE_ARCHIVE'] or 'local').lower()
    if mode == 'off':
        return NullTraceArchive()
    if mode == 'local':
        return LocalTraceArchive(env['TRACE_ARCHIVE_LOCAL_DIR'] or 'data/traces')
    if mode != 's3':
        raise ValueError('TRACE_ARCHIVE must be off, local or s3')
    bucket = env['TRACE_ARCHIVE_S3_BUCKET'] or env['DOCUMENTS_S3_BUCKET']
    if not bucket:
        raise ValueError('TRACE_ARCHIVE_S3_BUCKET or DOCUMENTS_S3_BUCKET is required in s3 archive mode')
    import boto3
    from botocore.config import Config
    session = boto3.Session(profile_name=env['AWS_PROFILE'] or None,
                            region_name=env['AWS_REGION'] or env['AWS_DEFAULT_REGION'] or 'us-east-1')
    client = session.client('s3', config=Config(connect_timeout=5, read_timeout=30,
                                                retries={'mode': 'standard', 'total_max_attempts': 3}))
    prefix = env['TRACE_ARCHIVE_S3_PREFIX'] or env['DOCUMENTS_S3_PREFIX_BASE'] \
        or 'content/health_insurance_document_agent/'
    return S3TraceArchive(client, bucket, prefix,
                          kms_key_id=env['TRACE_ARCHIVE_S3_KMS_KEY_ID'] or env['DOCUMENTS_S3_KMS_KEY_ID'] or None)


def write(archive, trace):
    """Archive one finished trace. A failure raises only when TRACE_ARCHIVE_REQUIRED=true."""
    try:
        archive.write(trace)
    except Exception as exc:
        if required():
            raise RuntimeError(f'Trace could not be archived: {type(exc).__name__}: {exc}') from exc
        logger.warning('Trace %s not archived: %s: %s', trace.get('trace_id'), type(exc).__name__, exc)


def document_summary(doc):
    """Identifiers of a document for span metadata; never its contents."""
    ref = getattr(doc, 'storage_ref', None) or {}
    return {'id': doc.id, 'name': doc.name, 'kind': doc.kind, 'manifest_key': ref.get('manifest_key'),
            'has_profile': bool(getattr(doc, 'profile', None)),
            'warnings_count': len(getattr(doc, 'warnings', []) or [])}


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
    """Which tools ran on which documents, and the locations they returned; never the text."""
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
