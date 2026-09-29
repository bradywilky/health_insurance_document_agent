from io import BytesIO
import json

import pytest

from backend.preprocessing.documents import ingest_document
from backend.storage.s3 import S3DocumentStore, configured_store
from backend.tools.documents import execute_document_tool


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.uploads = []
        self.reads = []
        self.fail_suffix = None

    def upload_fileobj(self, stream, bucket, key, ExtraArgs):
        if self.fail_suffix and key.endswith(self.fail_suffix):
            raise OSError('Upload failed')
        self.objects[(bucket, key)] = stream.read()
        self.uploads.append((key, ExtraArgs))

    def delete_object(self, *, Bucket, Key):
        self.objects.pop((Bucket, Key), None)

    def get_object(self, *, Bucket, Key):
        body = BytesIO(self.objects[(Bucket, Key)])
        self.reads.append(body)
        return {'Body':body}

    def get_paginator(self, operation):
        assert operation == 'list_objects_v2'
        return self

    def paginate(self, *, Bucket, Prefix):
        # One object per page exercises pagination beyond a single page.
        for bucket, key in sorted(self.objects):
            if bucket == Bucket and key.startswith(Prefix):
                yield {'Contents':[{'Key':key}]}


@pytest.fixture
def store():
    return S3DocumentStore(MemoryS3(), 'test-bucket', 'test-scope/')


def test_text_round_trip_and_manifest_published_last(store):
    doc = ingest_document('benefits.txt', b'Member copayment: USD 25.', store=store)
    reference = doc.storage_ref
    assert store.client.uploads[-1][0] == reference['manifest_key']
    assert any(key.endswith('/raw/benefits.txt') for key, _ in store.client.uploads)
    restored = store.load(reference['manifest_key'])
    assert (restored.id, restored.name, restored.blocks, restored.warnings) == (doc.id, doc.name, doc.blocks, doc.warnings)
    assert all(body.closed for body in store.client.reads)
    assert store.list_documents()[0]['document_id'] == doc.id


def test_table_round_trip_uses_real_s3_contract(store, insurance_workbook):
    doc = ingest_document(insurance_workbook.name, insurance_workbook.read_bytes(), store=store)
    restored = store.load(doc.storage_ref['manifest_key'])
    assert restored.table_inputs['s3_client'] is store.client
    result = execute_document_tool('table_tool', {'document_id':restored.id, 'name':'query_table',
        'parameters':{'sheet_name':'Claim Samples', 'filters':[{'column':'Claim ID','op':'eq','value':'0012'}]}},
        {restored.id:restored})
    assert result['table_result']['rows'][0]['Billed Amount'] == 220
    assert result['table_result']['rows'][0]['Claim ID'] == '0012'
    assert restored.table_metadata == doc.table_metadata
    assert all(body.closed for body in store.client.reads)
    assert any('/documents/preprocessed/insurance_mappings.xlsx/_metadata.json' in key for key, _ in store.client.uploads)


def test_failed_upload_never_publishes_manifest(store):
    store.client.fail_suffix = 'extracted.json'
    with pytest.raises(OSError, match='Upload failed'):
        ingest_document('rules.txt', b'Contract terms', store=store)
    assert store.list_documents() == []


def test_identical_uploads_use_same_filename_folder(store):
    first = ingest_document('rules.txt', b'Contract terms', store=store)
    second = ingest_document('rules.txt', b'Contract terms', store=store)
    assert first.id == second.id
    assert first.storage_ref == second.storage_ref
    assert len(store.list_documents()) == 1
    assert len(store.list_documents(limit=1)) == 1


def test_load_rejects_other_prefix_before_reading(store):
    with pytest.raises(ValueError, match='configured document prefix'):
        store.load('other/documents/' + 'a'*16 + '/' + 'b'*32 + '/manifest.json')
    assert not store.client.reads


def test_kms_override_and_default_bucket_encryption():
    client = MemoryS3()
    encrypted = S3DocumentStore(client, 'test-bucket', kms_key_id='test-key')
    ingest_document('rules.txt', b'Contract terms', store=encrypted)
    assert all(args['ServerSideEncryption'] == 'aws:kms' and args['SSEKMSKeyId'] == 'test-key'
               for _, args in client.uploads)
    client.uploads.clear()
    ingest_document('rules.txt', b'Other terms', store=S3DocumentStore(client, 'test-bucket'))
    assert all('ServerSideEncryption' not in args for _, args in client.uploads)


def test_local_mode_never_creates_aws_session(monkeypatch):
    monkeypatch.setenv('DOCUMENTS_STORAGE', 'local')
    monkeypatch.setenv('DOCUMENTS_S3_BUCKET', 'ignored-bucket')
    monkeypatch.setattr('backend.storage.s3.boto3.Session', lambda **kwargs: pytest.fail('No AWS call expected'))
    assert configured_store().bucket == "local"


def test_tampered_table_metadata_cannot_escape_saved_directory(store, insurance_workbook):
    doc = ingest_document(insurance_workbook.name, insurance_workbook.read_bytes(), store=store)
    key = doc.storage_ref['manifest_key'].replace('manifest.json', 'extracted.json')
    data = json.loads(store.client.objects[(store.bucket, key)])
    next(iter(data['table_metadata']['sheets'].values()))['csv_file'] = '../other.csv'
    store.client.objects[(store.bucket, key)] = json.dumps(data).encode()
    with pytest.raises(ValueError, match='CSV filename'):
        store.load(doc.storage_ref['manifest_key'])


def test_s3_mode_requires_explicit_bucket(monkeypatch):
    monkeypatch.setenv('DOCUMENTS_STORAGE', 's3')
    monkeypatch.delenv('DOCUMENTS_S3_BUCKET', raising=False)
    with pytest.raises(ValueError, match='required'):
        configured_store()


def test_replacement_failure_invalidates_previous_manifest(store):
    ingest_document('rules.txt', b'Old terms', store=store)
    store.client.fail_suffix = 'extracted.json'
    with pytest.raises(OSError):
        ingest_document('rules.txt', b'New terms', store=store)
    assert store.list_documents() == []


def test_replacement_removes_stale_outputs_and_preserves_other_files(store):
    old = ingest_document('rules.txt', b'Old terms', store=store)
    other = ingest_document('other.txt', b'Other terms', store=store)
    root = old.storage_ref['manifest_key'].removesuffix('manifest.json')
    store.client.objects[(store.bucket, root+'obsolete.csv')] = b'old'
    new = ingest_document('rules.txt', b'New terms', store=store)
    assert (store.bucket,root+'obsolete.csv') not in store.client.objects
    assert store.load(new.storage_ref['manifest_key']).blocks == new.blocks
    assert store.load(other.storage_ref['manifest_key']).blocks == other.blocks
    assert set(key.split('/')[2] for bucket,key in store.client.objects) == {'raw','preprocessed'}
    assert new.storage_ref['manifest_key'] == 'test-scope/documents/preprocessed/rules.txt/manifest.json'
