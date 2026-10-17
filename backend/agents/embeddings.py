"""Optional passage embeddings for semantic search, made once per document at upload.

Each text block is turned into a vector (a list of numbers that stands for its meaning) by a Bedrock embedding model,
Amazon Titan Text Embeddings V2 by default. Questions are embedded the same way at search time, and the blocks whose
vectors point the same way as the question's are its closest matches in meaning (backend/retrieval/semantic.py).
Off unless SEMANTIC_SEARCH=true (backend/config/settings.py).
"""
from base64 import b64decode, b64encode
from datetime import datetime, timezone
import json
import os

import numpy as np

from backend.observability import tracing
from backend.shared.parallel import parallel_map

EMBEDDINGS_VERSION = 1
DEFAULT_MODEL = 'amazon.titan-embed-text-v2:0'
DIMENSIONS = (256, 512, 1024)  # the sizes Titan Text Embeddings V2 returns
MAX_INPUT_CHARS = 20000  # well under Titan V2's 8,192-token limit; blocks are at most 1,600 characters


def encode(vector):
    """Base64 of float32 values: a quarter the size of JSON numbers, so a long document's vectors stay small."""
    return b64encode(np.asarray(vector, dtype='<f4').tobytes()).decode('ascii')


def decode(text):
    return np.frombuffer(b64decode(text), dtype='<f4')


class Embedder:
    """Embeds text with a Bedrock embedding model; model calls count toward usage_meter when given."""

    def __init__(self, model_id=None, dimensions=None, usage_meter=None):
        self.model_id = model_id or os.getenv('EMBEDDING_MODEL_ID') or DEFAULT_MODEL
        self.dimensions = dimensions or int(os.getenv('EMBEDDING_DIMENSIONS', '1024'))
        if self.dimensions not in DIMENSIONS:
            raise ValueError(f'EMBEDDING_DIMENSIONS must be one of {list(DIMENSIONS)}')
        self.usage_meter = usage_meter

    def _bedrock(self):
        if not hasattr(self, '_client'):
            import boto3
            from botocore.config import Config
            session = boto3.Session(profile_name=os.getenv('AWS_PROFILE') or None)
            self._client = session.client('bedrock-runtime',
                region_name=os.getenv('AWS_REGION') or session.region_name or 'us-east-1',
                config=Config(connect_timeout=10, read_timeout=60,
                              retries={'mode': 'adaptive',
                                       'total_max_attempts': int(os.getenv('BEDROCK_MAX_ATTEMPTS', '6'))}))
        return self._client

    def embed(self, text, *, purpose='Embedding'):
        """A unit-length vector for text (Titan normalizes it, so a dot product is the cosine similarity)."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Text to embed must be nonempty')
        body = {'inputText': text[:MAX_INPUT_CHARS], 'dimensions': self.dimensions, 'normalize': True}
        with tracing.span('embedding', purpose, model_name=self.model_id,
                          model_provider=tracing.MODEL_PROVIDER) as current:
            response = self._bedrock().invoke_model(modelId=self.model_id, body=json.dumps(body),
                                                    contentType='application/json', accept='application/json')
            reply = json.loads(response['body'].read())
            tokens = reply.get('inputTextTokenCount', 0)
            tracing.annotate(current, metadata={'dimensions': self.dimensions},
                             metrics={'input_tokens': tokens, 'total_tokens': tokens})
        if self.usage_meter is not None:
            self.usage_meter.add({'inputTokens': tokens})
        return reply['embedding']


def embed_document(doc, embedder, *, workers=None):
    """{'vectors': {block_id: encoded vector}, 'meta': ...} for every text block; one Bedrock call per block.

    Raises ValueError when no block could be embedded; failed blocks are listed in meta and left out of search.
    """
    if not doc.blocks:
        raise ValueError('No text blocks to embed')
    workers = workers or int(os.getenv('EMBEDDING_WORKERS', '8'))
    blocks = [b for b in doc.blocks if b['text'].strip()]
    vectors, failed = {}, []
    results = parallel_map(lambda b: embedder.embed(b['text'], purpose='Passage Embedding'), blocks, workers)
    for block, result in zip(blocks, results):
        if isinstance(result, Exception):
            failed.append({'block_id': block['block_id'], 'error': f'{type(result).__name__}: {result}'[:300]})
        else:
            vectors[block['block_id']] = encode(result)
    if not vectors:
        raise ValueError(f"No passages embedded: {failed[0]['error'] if failed else 'all blocks are empty'}")
    return {'vectors': vectors, 'meta': {
        'embeddings_version': EMBEDDINGS_VERSION, 'model_id': embedder.model_id,
        'dimensions': embedder.dimensions, 'vector_store': 'local',
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'blocks_total': len(doc.blocks), 'blocks_embedded': len(vectors), 'failed_blocks': failed}}


def make_embedder():
    """Callable for ingest_document(embed=...), using EMBEDDING_MODEL_ID and EMBEDDING_DIMENSIONS."""
    embedder = Embedder()
    return lambda doc: embed_document(doc, embedder)
