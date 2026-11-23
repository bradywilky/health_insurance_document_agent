"""Retrieval option 2: semantic search over text blocks and spreadsheet rows, then one model call.

Embeddings are faked; vectors point along a few made-up topics.
"""
import io
import math

import pandas as pd
import pytest

from backend.agents import semantic_context
from backend.agents.embeddings import embed_document, embed_rows
from backend.entrypoint import ask_question_local
from backend.preprocessing.documents import ingest_document
from backend.storage.filesystem import LocalDocumentStore

BENEFITS = (b'Emergency room services are covered after a USD 150 copay.\n\n'
            b'Ambulance transport requires prior authorization.')
TOPICS = [('emergency', 'er visit'), ('ambulance',), ('charge', 'billed', 'money')]


class FakeEmbedder:
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
    pd.DataFrame({'Source Field': ['claim_no', 'charge'], 'Canonical Field': ['claim_id', 'billed_amount']}) \
        .to_excel(out, sheet_name='Mappings', index=False)
    return out.getvalue()


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A text file with passage embeddings, a spreadsheet with row embeddings, and a text file with neither."""
    monkeypatch.delenv('VECTOR_STORE', raising=False)
    store, fake = LocalDocumentStore(tmp_path), FakeEmbedder()
    ingest_document('benefits.txt', BENEFITS, store=store, embed=lambda doc: embed_document(doc, fake))
    sheet = ingest_document('mappings.xlsx', workbook(), store=store)
    store.save_row_embeddings(sheet.storage_ref['manifest_key'], embed_rows(sheet, fake))
    ingest_document('notes.txt', b'Claims are paid monthly.', store=store)
    query = FakeEmbedder()
    monkeypatch.setattr(semantic_context, 'Embedder', lambda usage_meter=None: query)
    return {name: store.load(name) for name in ('benefits.txt', 'mappings.xlsx', 'notes.txt')}, query


def test_option_2_answers_from_text_and_row_chunks_in_one_call(library, native_script):
    docs, query = library
    requests = native_script(['The charge maps to billed_amount (sheet Mappings).'])
    response = ask_question_local('Which field holds the money charged?',
                                  documents=[docs['benefits.txt'], docs['mappings.xlsx']], retrieval_option=2)
    assert response['status'] == 'answered'
    [request] = requests  # one model call, no tools
    assert 'toolConfig' not in request
    prompt = request['messages'][0]['content'][0]['text']
    first = prompt.split('\n\n')[1]
    assert first.startswith('[1] mappings.xlsx | sheet Mappings, row 3 |') and 'billed_amount' in first
    assert '[3] benefits.txt' in prompt  # text blocks are searched too
    assert prompt.endswith('Based on the excerpts above, Which field holds the money charged?')
    assert query.calls == [('Query Embedding', 'Which field holds the money charged?')]  # embedded once
    assert response['protocol']['retrieval_option'] == 2 and response['protocol']['chunks'] == 4
    [source] = response['sources']
    assert source['tool'] == 'semantic_context'
    assert source['passages'][0] == {'filename': 'mappings.xlsx', 'location': 'sheet Mappings, row 3',
                                     'text': 'Sheet: Mappings (row 3)\nSource Field: charge\nCanonical Field: billed_amount'}


def test_option_2_reports_files_without_embeddings(library, native_script):
    docs, _ = library
    native_script(['ER visits cost USD 150.'])
    response = ask_question_local('How much is an ER visit?', documents=[docs['benefits.txt'], docs['notes.txt']],
                                  retrieval_option=2)
    assert response['status'] == 'answered'
    assert any('Not searched' in item and 'notes.txt' in item for item in response['limitations'])

    response = ask_question_local('How much is an ER visit?', documents=[docs['notes.txt']], retrieval_option=2)
    assert response['status'] == 'invalid_request' and 'upload portal' in response['message']


def test_upload_portal_generates_row_embeddings_for_spreadsheets(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('DOCUMENTS_STORAGE', 'local')
    monkeypatch.setenv('DOCUMENTS_LOCAL_DIR', str(tmp_path))
    monkeypatch.setenv('SEMANTIC_SEARCH', 'true')
    monkeypatch.setattr('backend.agents.embeddings.Embedder', lambda *a, **k: FakeEmbedder())
    store = LocalDocumentStore(tmp_path)
    key = ingest_document('mappings.xlsx', workbook(), store=store).storage_ref['manifest_key']
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / 'apps' / 'upload' / 'app.py')).run(timeout=30)
    next(s for s in app.selectbox if s.label == 'Saved document').set_value(key).run(timeout=30)
    assert any('No row embeddings' in c.value for c in app.caption)
    next(b for b in app.button if b.label == 'Generate embeddings').click().run(timeout=30)
    assert not app.exception
    assert store.load(key).row_embeddings['meta']['chunks_embedded'] == 2
    assert any('Row embeddings: 2 of 2 rows' in c.value for c in app.caption)
