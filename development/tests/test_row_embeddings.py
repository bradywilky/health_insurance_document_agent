"""Row embeddings: chunking a spreadsheet's data rows, saving row_embeddings.json, and the semantic_row_search tool.

Embeddings are faked; vectors point along a few made-up topics.
"""
import io
import math

import pandas as pd
import pytest

from backend.agents import document_agent
from backend.agents.document_agent import run_document_agent
from backend.agents.embeddings import embed_rows, row_chunks
from backend.preprocessing.documents import ingest_document
from backend.retrieval.semantic import row_search_documents
from backend.storage.filesystem import LocalDocumentStore

TOPICS = [('claim',), ('charge', 'billed', 'money'), ('member',)]


class FakeEmbedder:
    """Unit vectors over TOPICS (plus a 'none of these' axis), like a model that knows a charge is money."""
    model_id, dimensions = 'test-embed', 4

    def __init__(self):
        self.calls = []

    def embed(self, text, *, purpose='Embedding'):
        self.calls.append((purpose, text))
        lowered = text.casefold()
        vector = [float(any(word in lowered for word in words)) for words in TOPICS] + [0.1]
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]


def workbook():
    out = io.BytesIO()
    with pd.ExcelWriter(out) as writer:
        pd.DataFrame({'Notes': ['Mappings for the intake systems.', 'Identifiers keep leading zeros.']}) \
            .to_excel(writer, sheet_name='About', index=False)
        pd.DataFrame({'Source Field': ['claim_no', 'charge', 'member_no'],
                      'Canonical Field': ['claim_id', 'billed_amount', 'member_id'],
                      'Notes': ['Preserve zeros', 'USD', '']}).to_excel(writer, sheet_name='Mappings', index=False)
    return out.getvalue()


def saved(tmp_path, **options):
    store = LocalDocumentStore(tmp_path)
    doc = ingest_document('mappings.xlsx', workbook(), store=store)
    store.save_row_embeddings(doc.storage_ref['manifest_key'], embed_rows(doc, FakeEmbedder(), **options))
    return store, store.load('mappings.xlsx')


def test_row_chunks_cover_data_rows_with_their_sheet_and_excel_rows(tmp_path):
    doc = ingest_document('mappings.xlsx', workbook(), store=LocalDocumentStore(tmp_path))
    rows = row_chunks(doc)
    assert [(c['sheet'], c['rows']) for c in rows] == [('Mappings', [2, 2]), ('Mappings', [3, 3]), ('Mappings', [4, 4])]
    assert rows[1]['text'] == 'Sheet: Mappings (row 3)\nSource Field: charge\nCanonical Field: billed_amount\nNotes: USD'
    assert 'Notes:' not in rows[2]['text']  # empty cells are skipped
    [group] = row_chunks(doc, chunking='row_group', rows_per_chunk=5)
    assert group['rows'] == [2, 4] and group['text'].splitlines()[:2] == [
        'Sheet: Mappings (rows 2-4)', 'Source Field,Canonical Field,Notes']
    with pytest.raises(ValueError, match='chunking'):
        row_chunks(doc, chunking='page')


def test_saved_row_embeddings_find_rows_by_meaning(tmp_path):
    store, doc = saved(tmp_path)
    assert store.list_documents()[0]['has_row_embeddings']
    assert doc.row_embeddings['meta']['chunks_embedded'] == 3 and doc.row_embeddings['meta']['chunking'] == 'row'
    result = row_search_documents({doc.id: doc}, 'How much money was charged?', FakeEmbedder(), limit=1)
    [match] = result['matches']
    assert (match['sheet'], match['source_rows'], match['chunk_id']) == ('Mappings', [3, 3], 'R2')
    assert 'billed_amount' in match['text'] and 'not_indexed' not in result


def test_spreadsheets_without_matching_row_embeddings_are_reported_not_searched(tmp_path):
    _, doc = saved(tmp_path)
    other = FakeEmbedder()
    other.model_id = 'another-model'
    result = row_search_documents({doc.id: doc}, 'charge', other)
    assert result['matches'] == [] and 'another-model' in result['not_indexed'][0]['reason']
    plain = ingest_document('notes.txt', b'Claims are paid monthly.')
    bare = ingest_document('bare.xlsx', workbook())
    result = row_search_documents({plain.id: plain, bare.id: bare}, 'charge', FakeEmbedder())
    assert [d['filename'] for d in result['not_indexed']] == ['bare.xlsx']  # text files are not spreadsheets


def test_replacing_a_spreadsheet_drops_its_row_embeddings(tmp_path):
    store, _ = saved(tmp_path)
    ingest_document('mappings.xlsx', workbook(), store=store)
    assert store.load('mappings.xlsx').row_embeddings is None


def offered(request):
    return [t['toolSpec']['name'] for t in request['toolConfig']['tools']]


def test_agent_offers_semantic_row_search_only_for_spreadsheets_with_row_embeddings(tmp_path, native_script,
                                                                                    monkeypatch):
    embedder = FakeEmbedder()
    monkeypatch.setattr(document_agent, 'Embedder', lambda usage_meter=None: embedder)
    monkeypatch.delenv('SEMANTIC_SEARCH', raising=False)
    bare = ingest_document('bare.xlsx', workbook())
    requests = native_script([{'tool': 'table_tool', 'parameters': {'document_id': bare.id, 'name': 'list_sheets',
                                                                     'parameters': {}}},
                              {'tool': 'answer', 'parameters': {}}, 'The workbook has a Mappings sheet [E1].'])
    run_document_agent({bare.id: bare}, [bare.id], 'Which field holds the charge?')
    assert 'semantic_row_search' not in offered(requests[0])

    _, doc = saved(tmp_path)
    requests = native_script([{'tool': 'semantic_row_search', 'parameters': {'query': 'money charged', 'limit': 1}},
                              {'tool': 'answer', 'parameters': {}}, 'The charge maps to billed_amount [E1].'])
    result = run_document_agent({doc.id: doc}, [doc.id], 'Which field holds the charge?')
    assert 'semantic_row_search' in offered(requests[0])
    assert '- semantic_row_search:' in requests[0]['system'][0]['text']
    [evidence] = [e for e in result['evidence'] if e['tool'] == 'semantic_row_search']
    assert [m['source_rows'] for m in evidence['data']['matches']] == [[3, 3]]
    assert embedder.calls == [('Query Embedding', 'money charged')]
