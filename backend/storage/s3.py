"""S3 adapter and local/S3 document-store configuration."""
import os
import boto3
from botocore.config import Config
from backend.storage.documents import ObjectDocumentStore

class S3DocumentStore(ObjectDocumentStore):
    """Persist documents using an AWS S3 client."""


def configured_store(*, read_only=False):
    """Local mode makes no AWS calls. S3 mode uses profile credentials or deployment IAM roles."""
    mode = os.getenv('DOCUMENTS_STORAGE', 'local').strip().lower()
    if mode not in {'local', 's3'}:
        raise ValueError('DOCUMENTS_STORAGE must be local or s3')
    if mode == 'local':
        from backend.storage.filesystem import LocalDocumentStore
        return LocalDocumentStore(os.getenv("DOCUMENTS_LOCAL_DIR", "data/document_library"), read_only=read_only)
    bucket = os.getenv('DOCUMENTS_S3_BUCKET', '').strip()
    if not bucket:
        raise ValueError('DOCUMENTS_S3_BUCKET is required in s3 mode')
    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE') or None,
                            region_name=os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1')
    client = session.client('s3', config=Config(connect_timeout=5, read_timeout=60,
        retries={'mode':'standard', 'total_max_attempts':3}))
    return S3DocumentStore(client, bucket, os.getenv('DOCUMENTS_S3_PREFIX', 'document-agent/'),
                           kms_key_id=os.getenv('DOCUMENTS_S3_KMS_KEY_ID') or None, read_only=read_only)
