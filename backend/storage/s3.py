"""S3 adapter and local/S3 document-store configuration."""
from functools import lru_cache
import os
import boto3
from botocore.config import Config
from backend.storage.documents import ObjectDocumentStore, PREPROCESSED

PREFIX_BASE = 'content/health_insurance_document_agent/'


class S3DocumentStore(ObjectDocumentStore):
    """Persist documents using an AWS S3 client."""
    requires_group = True


def configured_store(*, read_only=False):
    """The document store the environment selects. Local mode makes no AWS calls. S3 mode uses profile credentials
    or deployment IAM roles.

    S3 documents live at s3://DOCUMENTS_S3_BUCKET/DOCUMENTS_S3_PREFIX_BASE<group>/DOCUMENTS_S3_PREFIX_PPDOCS/<filename>/.
    The group comes from DOCUMENTS_GROUP_NAME, or per request from ask_question(group_name=...).
    The store and its S3 client are reused while the settings stay the same (e.g. across a warm Lambda's requests).
    """
    names = ('DOCUMENTS_STORAGE', 'DOCUMENTS_GROUP_NAME', 'DOCUMENTS_S3_PREFIX_PPDOCS', 'DOCUMENTS_LOCAL_DIR',
             'DOCUMENTS_S3_BUCKET', 'DOCUMENTS_S3_PREFIX_BASE', 'DOCUMENTS_S3_KMS_KEY_ID', 'AWS_PROFILE',
             'AWS_REGION', 'AWS_DEFAULT_REGION')
    return _store(tuple((name, os.getenv(name)) for name in names), read_only)


@lru_cache(maxsize=8)
def _store(settings, read_only):
    env = dict(settings)
    mode = (env['DOCUMENTS_STORAGE'] or 'local').strip().lower()
    if mode not in {'local', 's3'}:
        raise ValueError('DOCUMENTS_STORAGE must be local or s3')
    group = (env['DOCUMENTS_GROUP_NAME'] or '').strip() or None
    preprocessed = (env['DOCUMENTS_S3_PREFIX_PPDOCS'] or '').strip() or PREPROCESSED
    if mode == 'local':
        from backend.storage.filesystem import LocalDocumentStore
        return LocalDocumentStore(env["DOCUMENTS_LOCAL_DIR"] or "data/document_library", group=group,
                                  preprocessed=preprocessed, read_only=read_only)
    bucket = (env['DOCUMENTS_S3_BUCKET'] or '').strip()
    if not bucket:
        raise ValueError('DOCUMENTS_S3_BUCKET is required in s3 mode')
    session = boto3.Session(profile_name=env['AWS_PROFILE'] or None,
                            region_name=env['AWS_REGION'] or env['AWS_DEFAULT_REGION'] or 'us-east-1')
    client = session.client('s3', config=Config(connect_timeout=5, read_timeout=60,
        retries={'mode':'standard', 'total_max_attempts':3}))
    return S3DocumentStore(client, bucket, env['DOCUMENTS_S3_PREFIX_BASE'] or PREFIX_BASE, group=group,
                           preprocessed=preprocessed, kms_key_id=env['DOCUMENTS_S3_KMS_KEY_ID'] or None,
                           read_only=read_only)
