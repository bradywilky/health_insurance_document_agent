"""Semantic search: the text blocks closest in meaning to a query, found by comparing embeddings
(backend/agents/embeddings.py) instead of shared words.

Where a saved document's vectors live is chosen by VECTOR_STORE:

    local      (default) in embeddings.json beside the document, compared in memory at search time. Needs no AWS
               service besides the Bedrock embedding model; suits development and small libraries.
    s3vectors  in an Amazon S3 Vectors index (S3_VECTORS_BUCKET, S3_VECTORS_INDEX). embeddings.json then keeps only
               the vector keys, so replacing or deleting the document can remove them.

Every vector in the index carries its group folder (`library`) and document_id as filterable metadata, and a query
is always filtered to the caller's group and selected documents. Block text is never stored in the index: a match
returns the block from the loaded document, so the index only routes search.
"""
from functools import lru_cache
import os

import numpy as np

from backend.agents.embeddings import decode

PUT_BATCH = 500  # S3 Vectors accepts at most 500 vectors per PutVectors call and 500 keys per DeleteVectors call


class S3VectorIndex:
    """One Amazon S3 Vectors index shared by every group; vectors are keyed and filtered by group folder."""
    name = 's3vectors'

    def __init__(self, client, bucket, index):
        if not bucket or not index:
            raise ValueError('S3_VECTORS_BUCKET and S3_VECTORS_INDEX are required when VECTOR_STORE=s3vectors')
        self.client, self.bucket, self.index = client, bucket, index

    def put(self, library, document_id, filename, vectors):
        """Store a document's vectors; returns their keys (library + document ID + block ID, unique per group)."""
        items = [{'key': f'{library}{document_id}/{block_id}', 'data': {'float32': decode(vector).tolist()},
                  'metadata': {'library': library, 'document_id': document_id, 'block_id': block_id,
                               'filename': filename}}
                 for block_id, vector in vectors.items()]
        for start in range(0, len(items), PUT_BATCH):
            self.client.put_vectors(vectorBucketName=self.bucket, indexName=self.index,
                                    vectors=items[start:start + PUT_BATCH])
        return [item['key'] for item in items]

    def delete(self, keys):
        for start in range(0, len(keys), PUT_BATCH):
            self.client.delete_vectors(vectorBucketName=self.bucket, indexName=self.index,
                                       keys=keys[start:start + PUT_BATCH])

    def query(self, library, document_ids, vector, top_k):
        """[(document_id, block_id, similarity)] among the given documents of one group, closest first."""
        reply = self.client.query_vectors(
            vectorBucketName=self.bucket, indexName=self.index, topK=top_k,
            queryVector={'float32': [float(x) for x in vector]}, returnMetadata=True, returnDistance=True,
            filter={'$and': [{'library': {'$eq': library}}, {'document_id': {'$in': list(document_ids)}}]})
        # The index is created with the cosine distance metric: similarity = 1 - distance.
        return [(hit['metadata']['document_id'], hit['metadata']['block_id'], 1 - hit['distance'])
                for hit in reply.get('vectors', [])]


def configured_vectors():
    """The S3 Vectors index VECTOR_STORE selects, or None for local vectors saved beside each document."""
    names = ('VECTOR_STORE', 'S3_VECTORS_BUCKET', 'S3_VECTORS_INDEX', 'AWS_PROFILE', 'AWS_REGION',
             'AWS_DEFAULT_REGION')
    return _vectors(tuple((name, os.getenv(name)) for name in names))


@lru_cache(maxsize=4)
def _vectors(settings):
    env = dict(settings)
    mode = (env['VECTOR_STORE'] or 'local').strip().lower()
    if mode not in {'local', 's3vectors'}:
        raise ValueError('VECTOR_STORE must be local or s3vectors')
    if mode == 'local':
        return None
    import boto3
    from botocore.config import Config
    session = boto3.Session(profile_name=env['AWS_PROFILE'] or None,
                            region_name=env['AWS_REGION'] or env['AWS_DEFAULT_REGION'] or 'us-east-1')
    client = session.client('s3vectors', config=Config(connect_timeout=5, read_timeout=60,
                                                       retries={'mode': 'standard', 'total_max_attempts': 3}))
    return S3VectorIndex(client, (env['S3_VECTORS_BUCKET'] or '').strip(), (env['S3_VECTORS_INDEX'] or '').strip())


def _unusable(doc, embedder, index):
    """Why doc's embeddings cannot be compared with this query, or None when they can."""
    meta = (doc.embeddings or {}).get('meta')
    if not meta:
        return 'no embeddings; generate them in the upload portal'
    if (meta.get('model_id'), meta.get('dimensions')) != (embedder.model_id, embedder.dimensions):
        return (f"embedded with {meta.get('model_id')} ({meta.get('dimensions')} dimensions), not "
                f'{embedder.model_id} ({embedder.dimensions}); regenerate its embeddings')
    if 'vectors' not in doc.embeddings and index is None:
        return 'its vectors are in an S3 Vectors index, but VECTOR_STORE is not s3vectors'
    return None


def _local_hits(doc, vector):
    vectors = doc.embeddings['vectors']
    ids = list(vectors)
    matrix = np.stack([decode(vectors[i]) for i in ids])
    # Titan returns unit-length vectors, so the dot product is the cosine similarity.
    return [(doc.id, block_id, float(score)) for block_id, score in zip(ids, matrix @ vector)]


def semantic_search_documents(documents, query, embedder, *, limit=8, index=None):
    """The blocks of the selected documents closest in meaning to query, most similar first.

    index: the S3 Vectors index for documents whose vectors live there (configured_vectors()); documents carrying
    their own vectors are compared in memory. Documents without usable embeddings are listed in not_indexed.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query must be a nonempty string')
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError('limit must be 1..20')
    not_indexed, local, remote = [], [], {}
    for doc in documents.values():
        reason = _unusable(doc, embedder, index)
        if reason:
            not_indexed.append({'document_id': doc.id, 'filename': doc.name, 'reason': reason})
        elif 'vectors' in doc.embeddings:
            local.append(doc)
        else:
            remote.setdefault(doc.embeddings['meta']['library'], []).append(doc.id)
    hits = []
    if local or remote:
        vector = np.asarray(embedder.embed(query, purpose='Query Embedding'), dtype='<f4')
        for doc in local:
            hits += _local_hits(doc, vector)
        for library, ids in remote.items():
            hits += index.query(library, ids, vector, limit)
    hits.sort(key=lambda hit: -hit[2])
    blocks = {doc.id: {b['block_id']: b for b in doc.blocks} for doc in documents.values()}
    matches = []
    for document_id, block_id, score in hits:
        block = blocks.get(document_id, {}).get(block_id)
        if block is not None:  # vectors of a replaced document version never match its current blocks
            matches.append({'document_id': document_id, 'filename': documents[document_id].name, **block,
                            'similarity': round(score, 4)})
        if len(matches) == limit:
            break
    output = {'matches': matches,
              'retrieval_method': 'semantic: text blocks ranked by closeness in meaning to the query (similarity '
                                  '0..1). The closest blocks are returned even when none is relevant, so judge each '
                                  'one. Spreadsheet data rows are not embedded; use table tools for them.',
              'warnings': {doc.name: doc.warnings for doc in documents.values() if doc.warnings}}
    if not_indexed:
        output['not_indexed'] = not_indexed
    return output
