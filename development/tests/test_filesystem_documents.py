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
    sheet = next(iter(loaded.table_metadata['sheets'].values()))
    assert (tmp_path / 'documents/preprocessed/claims.csv' / sheet['csv_file']).read_bytes().count(b'0012') == 1
    assert loaded.table_metadata == doc.table_metadata
    frame = load_table(loaded.table_folder, sheet)
    assert sum(frame.column("Amount")) == 320
    assert frame.rows[0]["Claim"] == "0012"


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


def test_delete_removes_the_document_and_its_original(tmp_path):
    store = LocalDocumentStore(tmp_path)
    ingest_document('claims.csv', b'Claim,Amount\n0012,220\n', store=store)
    ingest_document('keep.txt', b'Other', store=store)
    store.delete('claims.csv')
    assert [d['filename'] for d in store.list_documents()] == ['keep.txt']
    assert not [p for p in (tmp_path / 'documents/preprocessed/claims.csv').rglob('*') if p.is_file()]
    assert not (tmp_path / 'documents/raw/claims.csv').exists()
    assert (tmp_path / 'documents/raw/keep.txt').exists()
    store.delete('claims.csv')  # already gone: nothing to do
    with pytest.raises(PermissionError):
        LocalDocumentStore(tmp_path, read_only=True).delete('keep.txt')
    with pytest.raises(ValueError):
        store.delete('../keep.txt')
