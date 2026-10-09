"""S3 adapter and local/S3 document-store configuration."""
import os
import boto3
from botocore.config import Config
from backend.storage.documents import ObjectDocumentStore, PREPROCESSED

PREFIX_BASE = 'content/health_insurance_document_agent/'


class S3DocumentStore(ObjectDocumentStore):
    """Persist documents using an AWS S3 client."""
    requires_group = True


def configured_store(*, read_only=False):
    """Local mode makes no AWS calls. S3 mode uses profile credentials or deployment IAM roles.

    S3 documents live at s3://DOCUMENTS_S3_BUCKET/DOCUMENTS_S3_PREFIX_BASE<group>/DOCUMENTS_S3_PREFIX_PPDOCS/<filename>/.
    The group comes from DOCUMENTS_GROUP_NAME, or per request from ask_question(group_name=...).
    """
    mode = os.getenv('DOCUMENTS_STORAGE', 'local').strip().lower()
    if mode not in {'local', 's3'}:
        raise ValueError('DOCUMENTS_STORAGE must be local or s3')
    group = os.getenv('DOCUMENTS_GROUP_NAME', '').strip() or None
    preprocessed = os.getenv('DOCUMENTS_S3_PREFIX_PPDOCS', '').strip() or PREPROCESSED
    if mode == 'local':
        from backend.storage.filesystem import LocalDocumentStore
        return LocalDocumentStore(os.getenv("DOCUMENTS_LOCAL_DIR", "data/document_library"), group=group,
                                  preprocessed=preprocessed, read_only=read_only)
    bucket = os.getenv('DOCUMENTS_S3_BUCKET', '').strip()
    if not bucket:
        raise ValueError('DOCUMENTS_S3_BUCKET is required in s3 mode')
    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE') or None,
                            region_name=os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1')
    client = session.client('s3', config=Config(connect_timeout=5, read_timeout=60,
        retries={'mode':'standard', 'total_max_attempts':3}))
    return S3DocumentStore(client, bucket, os.getenv('DOCUMENTS_S3_PREFIX_BASE', PREFIX_BASE), group=group,
                           preprocessed=preprocessed, kms_key_id=os.getenv('DOCUMENTS_S3_KMS_KEY_ID') or None,
                           read_only=read_only)
