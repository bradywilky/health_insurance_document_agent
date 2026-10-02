import pytest
from backend.storage.filesystem import LocalDocumentStore
from backend.preprocessing.documents import ingest_document


def test_persistent_text_and_original(tmp_path):
    writer = LocalDocumentStore(tmp_path)
    doc = ingest_document('rules.txt', b'Rate USD 100', store=writer)
    reader = LocalDocumentStore(tmp_path, read_only=True)
    loaded = reader.load(doc.storage_ref['manifest_key'])
    assert loaded.blocks == doc.blocks
    assert (tmp_path / 'documents/raw/rules.txt').read_bytes() == b'Rate USD 100'
    assert len(reader.list_documents()) == 1
    with pytest.raises(PermissionError):
        reader.save(doc, b'Rate USD 100')
    with pytest.raises(PermissionError):
        reader.client.delete_object(Bucket='local', Key='documents/raw/rules.txt')


def test_persistent_table_roundtrip(tmp_path):
    doc = ingest_document('claims.csv', b'Claim,Amount\n0012,220\n0013,100\n', store=LocalDocumentStore(tmp_path))
    reader = LocalDocumentStore(tmp_path, read_only=True)
    loaded = reader.load(doc.storage_ref['manifest_key'])
    from backend.storage.tables import load_table
    # Read the persisted CSV via the same object interface consumed by table tools.
    inputs = loaded.table_inputs
    sheet = next(iter(loaded.table_metadata['sheets'].values()))
    key = 'documents/preprocessed/claims.csv/' + sheet['csv_file']
    with inputs['s3_client'].get_object(Bucket='local', Key=key)['Body'] as body:
        assert b'0012' in body.read()
    assert loaded.table_metadata == doc.table_metadata
    frame = load_table(**inputs, sheet_meta=sheet)
    assert frame["Amount"].sum() == 320
    assert str(frame.iloc[0]["Claim"]) == "0012"


@pytest.mark.parametrize('key', ['../escape', '/absolute', 'documents/../../escape', 'C:/escape', 'documents\\escape'])
def test_local_keys_cannot_escape(tmp_path, key):
    with pytest.raises(ValueError):
        LocalDocumentStore(tmp_path).client.get_object(Bucket='local', Key=key)


def test_failed_replacement_not_listed(tmp_path, monkeypatch):
    store = LocalDocumentStore(tmp_path)
    ingest_document('policy.txt', b'Old', store=store)
    def fail(*args):
        raise OSError('Disk unavailable')
    monkeypatch.setattr(store, '_json', fail)
    with pytest.raises(OSError):
        ingest_document('policy.txt', b'New', store=store)
    assert LocalDocumentStore(tmp_path).list_documents() == []
