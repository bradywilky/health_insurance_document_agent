"""Real file parsers/storage/tools; scripted model calls do not grade model reasoning."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.document_agent import run_document_agent
from backend.preprocessing.documents import ingest_document, MAX_FILE_BYTES
from backend.storage.filesystem import LocalDocumentStore
from backend.storage.s3 import S3DocumentStore
from backend.tools.documents import execute_document_tool
from backend.retrieval.search import search_documents
from development.scripts.make_robustness_samples import create_samples
from development.tests.test_s3_documents import MemoryS3


@pytest.fixture(scope='module')
def pack(tmp_path_factory):
    return create_samples(tmp_path_factory.mktemp('robustness')/'pack')


@pytest.fixture(params=['local', 's3-contract'])
def library(request, tmp_path):
    return (LocalDocumentStore(tmp_path) if request.param == 'local'
            else S3DocumentStore(MemoryS3(), 'test-bucket', 'robustness/'))


def load_file(pack, name, store=None):
    return ingest_document(name, (pack/name).read_bytes(), store=store)


def query(doc, sheet, **parameters):
    return execute_document_tool('table_tool', {'document_id': doc.id, 'name': 'query_table',
        'parameters': {'sheet_name': sheet, **parameters}}, {doc.id: doc})


def test_pack_upload_outcomes_and_warning_persistence(pack, library):
    saved = 0
    for case in json.loads((pack/'upload_expectations.json').read_text()):
        if case['expected'] == 'rejected':
            before = library.list_documents()
            with pytest.raises(Exception):
                load_file(pack, case['file'], library)
            assert library.list_documents() == before
        else:
            doc = load_file(pack, case['file'], library)
            restored = library.load(doc.storage_ref['manifest_key'])
            assert restored.warnings == doc.warnings
            assert restored.blocks == doc.blocks
            for fragment in case['warning_fragments']:
                assert any(fragment in warning for warning in restored.warnings), case['file']
            saved += 1
    assert len(library.list_documents()) == saved


def test_truncated_workbook_retains_late_rows_sheets_and_complete_totals(pack, library):
    uploaded = load_file(pack, 'large_crosswalk.xlsx', library)
    doc = library.load(uploaded.storage_ref['manifest_key'])
    assert len(doc.blocks) == 2000
    assert not search_documents({doc.id: doc}, 'TAILMARKER')['matches']
    tail = query(doc, 'Crosswalk', filters=[{'column': 'Field ID', 'op': 'eq', 'value': 'F02105'}])
    assert tail['table_result']['rows'][0]['Definition'] == 'TAILMARKER'
    assert any('2000' in w for w in tail['warnings'])
    assert query(doc, 'Late Dictionary')['table_result']['rows'][0]['Meaning'] == 'Late sheet definition'
    totals = query(doc, 'Crosswalk', aggregations=[{'op': 'count_rows', 'as': 'records'},
                   {'column': 'Units', 'op': 'sum', 'as': 'units'}])['table_result']
    assert totals['rows'] == [{'records': 2105, 'units': '2105'}]


def test_uncached_formula_is_missing_not_zero(pack):
    doc = load_file(pack, 'formula_caches.xlsx')
    rows = query(doc, 'Calculations')['table_result']['rows']
    amounts = {row['Record']: row['Amount'] for row in rows}
    assert amounts == {'cached': 176.0, 'uncached': None, 'zero': 0.0}
    result = query(doc, 'Calculations', aggregations=[{'column': 'Amount', 'op': 'sum', 'as': 'known'}])['table_result']
    assert result['rows'] == [{'known': '176.0'}]
    assert result['null_counts']['Amount'] == 1


def test_pdf_and_word_omissions_are_real(pack):
    mixed = load_file(pack, 'mixed_scan.pdf')
    assert [b['location']['page'] for b in mixed.blocks] == [1, 3]
    assert 'VIOLET-77' not in str(mixed.blocks)
    capped = load_file(pack, 'page_limit.pdf')
    assert max(b['location']['page'] for b in capped.blocks) == 200
    assert 'ORCHID-201' not in str(capped.blocks)
    word = load_file(pack, 'body_and_headers.docx')
    assert 'AZURE-19' not in str(word.blocks)
    assert 'Cedar Team' not in str(word.blocks)
    assert any(b['location'].get('table') == 1 and b['location'].get('row') == 2 for b in word.blocks)
    assert all('page' not in b['location'] for b in word.blocks)


def test_table_only_answer_receives_extraction_warnings(pack, native_script):
    doc = load_file(pack, 'large_crosswalk.xlsx')
    requests = native_script([
        {'tool': 'table_tool', 'parameters': {'document_id': doc.id, 'name': 'query_table',
         'parameters': {'sheet_name': 'Crosswalk', 'filters': [{'column': 'Field ID', 'op': 'eq', 'value': 'F02105'}]}}},
        {'tool': 'answer', 'parameters': {}}, 'Definition: TAILMARKER [E1].'])
    result = run_document_agent({doc.id: doc}, [doc.id], 'Define F02105')
    payload = json.loads(requests[-1]['messages'][0]['content'][0]['text'])
    assert any('2000 text blocks' in w for w in payload['limitations'])
    assert any('2000 text blocks' in w for w in result['limitations'])
    assert result['evidence'][0]['data']['table_result']['rows'][0]['Definition'] == 'TAILMARKER'


def test_unread_source_warnings_reach_answer_writer(pack, native_script):
    scan = load_file(pack, 'scan_only.pdf')
    text = ingest_document('text.txt', b'Base rate is USD 88.')
    requests = native_script([{'tool': 'read_document', 'parameters': {'document_id': text.id}},
                             {'tool': 'answer', 'parameters': {}}, 'Base rate USD 88 [E1]; scan unavailable.'])
    docs = {doc.id: doc for doc in [scan, text]}
    result = run_document_agent(docs, list(docs), 'Rate and exceptions?')
    payload = json.loads(requests[-1]['messages'][0]['content'][0]['text'])
    assert any('scan_only.pdf' in w and 'OCR' in w for w in payload['limitations'])
    assert any('No searchable text' in w for w in result['limitations'])


def test_scan_only_fallback_preserves_ocr_warning(pack, native_script):
    doc = load_file(pack, 'scan_only.pdf')
    native_script([{'tool': 'read_document', 'parameters': {'document_id': doc.id}}] +
                  [{'tool': 'answer', 'parameters': {}}]*3)
    result = run_document_agent({doc.id: doc}, [doc.id], 'What is the exception?')
    assert 'could not retrieve supporting content' in result['answer']
    assert any('OCR' in w for w in result['limitations'])


@pytest.mark.parametrize('name,content,match', [
    ('empty.txt', b'', 'nonempty'), ('large.txt', b'x'*(MAX_FILE_BYTES+1), '20 MB'),
    ('program.exe', b'not supported', 'Unsupported'),
], ids=['empty', 'oversize', 'unsupported'])
def test_invalid_inputs_never_reach_storage(name, content, match):
    class NeverSave:
        def save(self, *args):
            pytest.fail('Invalid input reached storage')
    with pytest.raises(ValueError, match=match):
        ingest_document(name, content, store=NeverSave())


def test_same_filename_replacement_and_distinct_versions(pack, library):
    old = ingest_document('same_name.txt', (pack/'replacement_v1/same_name.txt').read_bytes(), store=library)
    new = ingest_document('same_name.txt', (pack/'replacement_v2/same_name.txt').read_bytes(), store=library)
    assert old.id != new.id
    assert len(library.list_documents()) == 1
    assert '99' in str(library.load(old.storage_ref['manifest_key']).blocks)
    ingest_document('same_name_v1.txt', (pack/'replacement_v1/same_name.txt').read_bytes(), store=library)
    assert len(library.list_documents()) == 2


def test_evaluation_loads_only_question_inputs(pack, tmp_path, monkeypatch):
    from development.evaluation import documents as evaluation
    monkeypatch.setattr(evaluation, 'load_dotenv', lambda: None)
    monkeypatch.delenv('BEDROCK_MODEL_ID', raising=False)
    calls = []
    def fake_run(documents, ids, question, **kwargs):
        calls.append(([documents[id].name for id in ids], question))
        return {'answer': 'Review required', 'evidence': [], 'limitations': []}
    monkeypatch.setattr(evaluation, 'run_document_agent', fake_run)
    output = tmp_path/'evaluation.jsonl'
    monkeypatch.setattr('sys.argv', ['evaluation', '--fixtures', str(pack), '--case', 'tail-row', '--output', str(output)])
    evaluation.main()
    assert calls == [(['large_crosswalk.xlsx'], 'What is the definition of F02105?')]
    record = json.loads(output.read_text())
    assert record['status'] == 'needs_human_review'
    assert 'TAILMARKER' in record['expected']


@pytest.mark.parametrize('count', [3, 11])
def test_upload_batch_failure_isolation_and_limit(pack, tmp_path, monkeypatch, count):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('DOCUMENTS_STORAGE', 'local')
    monkeypatch.setenv('DOCUMENTS_LOCAL_DIR', str(tmp_path))
    names = ['instruction_trap.txt', 'broken.pdf', 'insurance_claims.csv'] if count == 3 else [f'file{i}.txt' for i in range(11)]
    uploaded = [SimpleNamespace(name=name, getvalue=lambda name=name: (pack/name).read_bytes() if count == 3 else b'content') for name in names]
    monkeypatch.setattr('streamlit.file_uploader', lambda *args, **kwargs: uploaded)
    path = Path(__file__).resolve().parents[2]/'apps/upload/app.py'
    app = AppTest.from_file(str(path)).run(timeout=30)
    next(b for b in app.button if b.label == 'Process and save files').click().run(timeout=30)
    assert not app.exception
    if count == 3:
        assert len(app.success) == 2
        assert any('Could not save broken.pdf' in e.value for e in app.error)
        assert len(LocalDocumentStore(tmp_path).list_documents()) == 2
    else:
        assert any('at most 10' in e.value for e in app.error)
        assert LocalDocumentStore(tmp_path).list_documents() == []
