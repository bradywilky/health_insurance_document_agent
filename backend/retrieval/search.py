"""Lexical retrieval over selected preprocessed documents."""
import re


def search_documents(documents, query, *, limit=8):
    """Small-corpus lexical retrieval, restricted to the supplied selected documents."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query must be a nonempty string')
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError('limit must be 1..20')
    stop = {'the','a','an','is','are','what','which','of','to','for','in','and','or','does','do','with'}
    terms = set(re.findall(r'[\w]+', query.casefold())) - stop
    hits = []
    for doc in documents.values():
        for block in doc.blocks:
            text = block['text'].casefold()
            words = set(re.findall(r'[\w]+', text))
            score = len(terms & words)
            if query.casefold() in text:
                score += len(terms) + 1
            if score:
                hits.append({'document_id':doc.id, 'filename':doc.name, **block, 'score':score})
    hits.sort(key=lambda hit: (-hit['score'], hit['filename'], hit['block_id']))
    return {'matches':hits[:limit], 'total_matching_blocks':len(hits),
            'truncated':len(hits)>limit,
            'retrieval_method':'lexical; no semantic matching or inferred policy precedence',
            'warnings':{doc.name:doc.warnings for doc in documents.values() if doc.warnings}}
