"""Semantic search: upload-time passage embeddings, local and S3 Vectors storage, and the semantic_search tool.

Embeddings and the S3 Vectors service are faked; vectors point along a few made-up topics.
"""
import io
import json
import math

import pytest

from backend.agents import document_agent
from backend.agents.document_agent import run_document_agent
from backend.agents.document_protocol import examined_documents
from backend.agents.embeddings import Embedder, embed_document
from backend.agents.llm import Usage
from backend.preprocessing.documents import ingest_document
from backend.retrieval.semantic import S3VectorIndex, semantic_search_documents
from backend.storage.filesystem import LocalDocumentStore

BENEFITS = (b'Emergency room services are covered after a USD 150 copay.\n\n'
            b'Ambulance transport requires prior authorization.\n\n'
            b'Coverage ends on the last day of the month after termination.')
TOPICS = [('emergency', 'er visit'), ('ambulance',), ('coverage ends', 'cancel')]


class FakeEmbedder:
    """Unit vectors over TOPICS (plus a 'none of these' axis), like a model that knows ER means emergency room."""
    model_id, dimensions = 'test-embed', 4

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def embed(self, text, *, purpose='Embedding'):
        self.calls.append((purpose, text))
        if self.fail_on and self.fail_on in text:
            raise ValueError('throttled')
        lowered = text.casefold()
        vector = [float(any(word in lowered for word in words)) for words in TOPICS] + [0.1]
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]


class FakeS3Vectors:
    """The PutVectors, DeleteVectors and QueryVectors calls of an S3 Vectors client, with $and/$eq/$in filters."""

    def __init__(self):
        self.vectors, self.queries = {}, []

    def put_vectors(self, *, vectorBucketName, indexName, vectors):
        assert (vectorBucketName, indexName) == ('vector-bucket', 'passages') and len(vectors) <= 500
        self.vectors.update({v['key']: v for v in vectors})

    def delete_vectors(self, *, vectorBucketName, indexName, keys):
        for key in keys:
            self.vectors.pop(key, None)

    def query_vectors(self, *, vectorBucketName, indexName, topK, queryVector, filter, returnMetadata,
                      returnDistance):
        self.queries.append(filter)
        def matches(meta, rule):
            if '$and' in rule:
                return all(matches(meta, r) for r in rule['$and'])
            [(field, test)] = rule.items()
            return meta[field] == test['$eq'] if '$eq' in test else meta[field] in test['$in']
        query = queryVector['float32']
        hits = [{'key': key, 'metadata': v['metadata'],
                 'distance': 1 - sum(a * b for a, b in zip(query, v['data']['float32']))}
                for key, v in self.vectors.items() if matches(v['metadata'], filter)]
        return {'vectors': sorted(hits, key=lambda h: h['distance'])[:topK]}


def embedded(name='benefits.txt', content=BENEFITS, **kwargs):
    return ingest_document(name, content, embed=lambda d: embed_document(d, FakeEmbedder()), **kwargs)


def test_semantic_search_finds_blocks_by_meaning_from_saved_local_vectors(tmp_path):
    store = LocalDocumentStore(tmp_path)
    doc = embedded(store=store)
    assert doc.embeddings['meta']['blocks_embedded'] == 3 and doc.embeddings['meta']['model_id'] == 'test-embed'
    loaded = store.load(doc.storage_ref['manifest_key'])
    assert loaded.embeddings == doc.embeddings and set(loaded.embeddings['vectors']) == {'B1', 'B2', 'B3'}
    embedder = FakeEmbedder()
    result = semantic_search_documents({loaded.id: loaded}, 'ER visit cost?', embedder, limit=2)
    assert [m['block_id'] for m in result['matches']] == ['B1', 'B2']
    assert result['matches'][0]['text'].startswith('Emergency room') and result['matches'][0]['similarity'] > 0.9
    assert embedder.calls == [('Query Embedding', 'ER visit cost?')] and 'not_indexed' not in result
    assert examined_documents([{'tool': 'semantic_search', 'parameters': {}, 'data': result}]) == {loaded.id}


def test_documents_without_matching_embeddings_are_reported_not_searched():
    plain = ingest_document('plain.txt', b'Dental cleanings twice a year.')
    doc = embedded()
    other_model = FakeEmbedder()
    other_model.model_id = 'another-model'
    result = semantic_search_documents({plain.id: plain, doc.id: doc}, 'ambulance', other_model)
    assert result['matches'] == [] and other_model.calls == []  # nothing searchable, so no query embedding
    reasons = {item['filename']: item['reason'] for item in result['not_indexed']}
    assert 'no embeddings' in reasons['plain.txt'] and 'another-model' not in reasons['plain.txt']
    assert 'regenerate its embeddings' in reasons['benefits.txt']
    with pytest.raises(ValueError, match='limit'):
        semantic_search_documents({doc.id: doc}, 'ambulance', FakeEmbedder(), limit=21)


def test_failed_block_embeddings_are_listed_and_never_block_an_upload():
    doc = ingest_document('benefits.txt', BENEFITS,
                          embed=lambda d: embed_document(d, FakeEmbedder(fail_on='Ambulance'), workers=2))
    assert set(doc.embeddings['vectors']) == {'B1', 'B3'}
    assert doc.embeddings['meta']['failed_blocks'] == [{'block_id': 'B2', 'error': 'ValueError: throttled'}]
    failed = ingest_document('benefits.txt', BENEFITS,
                             embed=lambda d: embed_document(d, FakeEmbedder(fail_on=' ')))
    assert failed.embeddings is None
    assert any(w.startswith('Embeddings not generated: ValueError: No passages embedded') for w in failed.warnings)


def test_s3_vectors_hold_the_vectors_scoped_by_group_and_follow_replacements_and_deletes(tmp_path):
    client = FakeS3Vectors()
    index = S3VectorIndex(client, 'vector-bucket', 'passages')
    shared = LocalDocumentStore(tmp_path, vectors=index)
    acme, other = shared.for_group('acme.com'), shared.for_group('other.org')
    doc = embedded(store=acme)
    embedded(store=other)  # the same file in another group: same document ID, separate vectors
    assert sorted(client.vectors) == sorted([f'acme.com/{doc.id}/B{n}' for n in (1, 2, 3)]
                                            + [f'other.org/{doc.id}/B{n}' for n in (1, 2, 3)])
    assert client.vectors[f'acme.com/{doc.id}/B2']['metadata'] == {
        'library': 'acme.com/', 'document_id': doc.id, 'block_id': 'B2', 'filename': 'benefits.txt'}
    loaded = acme.load('benefits.txt')
    assert 'vectors' not in loaded.embeddings and len(loaded.embeddings['keys']) == 3
    assert loaded.embeddings['meta']['library'] == 'acme.com/'
    result = semantic_search_documents({loaded.id: loaded}, 'ambulance', FakeEmbedder(), limit=1, index=index)
    assert [m['block_id'] for m in result['matches']] == ['B2'] and result['matches'][0]['similarity'] > 0.9
    assert client.queries[-1] == {'$and': [{'library': {'$eq': 'acme.com/'}},
                                           {'document_id': {'$in': [loaded.id]}}]}
    # Without the index configured, those documents cannot be searched.
    assert 'not s3vectors' in semantic_search_documents({loaded.id: loaded}, 'ambulance',
                                                        FakeEmbedder())['not_indexed'][0]['reason']
    # Saving a new version under the same filename removes the old version's vectors.
    new = embedded(content=BENEFITS + b'\n\nDental is excluded.', store=acme)
    assert sorted(k for k in client.vectors if k.startswith('acme.com/')) == \
        sorted(f'acme.com/{new.id}/B{n}' for n in (1, 2, 3, 4))
    acme.delete('benefits.txt')
    assert all(k.startswith('other.org/') for k in client.vectors) and len(client.vectors) == 3


def test_a_vector_index_failure_keeps_the_upload_without_embeddings(tmp_path):
    class Down(FakeS3Vectors):
        def put_vectors(self, **kwargs):
            raise ConnectionError('index unreachable')
    store = LocalDocumentStore(tmp_path, vectors=S3VectorIndex(Down(), 'vector-bucket', 'passages'))
    embedded(store=store)
    loaded = store.load('benefits.txt')
    assert loaded.embeddings is None
    assert 'Embeddings not saved: ConnectionError: index unreachable' in loaded.warnings


def test_save_embeddings_adds_them_to_a_saved_document(tmp_path):
    store = LocalDocumentStore(tmp_path)
    doc = ingest_document('benefits.txt', BENEFITS, store=store)
    assert store.load('benefits.txt').embeddings is None
    store.save_embeddings(doc.storage_ref['manifest_key'], embed_document(doc, FakeEmbedder()))
    assert store.list_documents()[0]['has_embeddings']
    assert set(store.load('benefits.txt').embeddings['vectors']) == {'B1', 'B2', 'B3'}


def test_bedrock_embedder_sends_titan_requests_and_counts_usage(monkeypatch):
    class Bedrock:
        def __init__(self):
            self.requests = []

        def invoke_model(self, **kwargs):
            self.requests.append(kwargs)
            return {'body': io.BytesIO(json.dumps({'embedding': [0.6, 0.8], 'inputTextTokenCount': 7}).encode())}
    monkeypatch.delenv('EMBEDDING_MODEL_ID', raising=False)
    usage = Usage()
    embedder = Embedder(dimensions=256, usage_meter=usage)
    embedder._client = Bedrock()
    assert embedder.embed('ER visit') == [0.6, 0.8]
    [request] = embedder._client.requests
    assert request['modelId'] == 'amazon.titan-embed-text-v2:0'
    assert json.loads(request['body']) == {'inputText': 'ER visit', 'dimensions': 256, 'normalize': True}
    assert usage.totals() == {'llm_calls': 1, 'input_tokens': 7, 'output_tokens': 0}
    with pytest.raises(ValueError, match='EMBEDDING_DIMENSIONS'):
        Embedder(dimensions=300)


def read_and_answer(doc):
    return [{'tool': 'read_document', 'parameters': {'document_id': doc.id}}, {'tool': 'answer', 'parameters': {}},
            f"{doc.blocks[0]['text']} [E1]."]


def offered(request):
    return [t['toolSpec']['name'] for t in request['toolConfig']['tools']]


def test_agent_offers_semantic_search_only_when_enabled_and_documents_are_embedded(native_script, monkeypatch):
    doc = embedded()
    embedder = FakeEmbedder()
    monkeypatch.setattr(document_agent, 'Embedder', lambda usage_meter=None: embedder)
    monkeypatch.delenv('SEMANTIC_SEARCH', raising=False)
    requests = native_script(read_and_answer(doc))
    run_document_agent({doc.id: doc}, [doc.id], 'Is an ER visit covered?')
    assert 'semantic_search' not in offered(requests[0])
    assert 'semantic_search' not in requests[0]['system'][0]['text']

    monkeypatch.setenv('SEMANTIC_SEARCH', 'true')
    plain = ingest_document('plain.txt', b'Dental cleanings twice a year.')
    requests = native_script(read_and_answer(plain))
    run_document_agent({plain.id: plain}, [plain.id], 'Is an ER visit covered?')
    assert 'semantic_search' not in offered(requests[0])

    requests = native_script([{'tool': 'semantic_search', 'parameters': {'query': 'ER visit', 'limit': 1}},
                              {'tool': 'answer', 'parameters': {}},
                              'Emergency room services are covered after a USD 150 copay [E1].'])
    result = run_document_agent({doc.id: doc}, [doc.id], 'Is an ER visit covered?')
    assert 'semantic_search' in offered(requests[0]) and '- semantic_search:' in requests[0]['system'][0]['text']
    [evidence] = [e for e in result['evidence'] if e['tool'] == 'semantic_search']
    assert [m['block_id'] for m in evidence['data']['matches']] == ['B1']
    assert embedder.calls == [('Query Embedding', 'ER visit')]
