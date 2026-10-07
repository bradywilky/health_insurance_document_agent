"""Lexical retrieval over selected preprocessed documents."""
from collections import OrderedDict
import re

from backend.shared.rows import row_location, row_text

STOP = {'the','a','an','is','are','what','which','of','to','for','in','and','or','does','do','with'}
_FRAMES = OrderedDict()  # (document id, sheet) -> Table; small LRU so repeat searches are fast
_FRAME_CACHE_SIZE = 16


def _frame(doc, sheet, meta):
    from backend.storage.tables import load_table
    key = (doc.id, sheet)
    if key not in _FRAMES:
        _FRAMES[key] = load_table(**doc.table_inputs, sheet_name=sheet, sheet_meta=meta)
        if len(_FRAMES) > _FRAME_CACHE_SIZE:
            _FRAMES.popitem(last=False)
    _FRAMES.move_to_end(key)
    return _FRAMES[key]


def _units(doc):
    """Searchable units: text blocks, plus every row of each table data sheet read from its CSV."""
    data_sheets = {}
    if doc.table_inputs and doc.table_metadata:
        data_sheets = {name: meta for name, meta in doc.table_metadata.get('sheets', {}).items()
                       if meta.get('sheet_type') != 'metadata'}
    for block in doc.blocks:
        location = block.get('location', {})
        # Preview rows of data sheets are covered by the full-row scan below.
        if not ('csv_record' in location and location.get('sheet') in data_sheets):
            yield block
    for sheet, meta in data_sheets.items():
        for index, row in enumerate(_frame(doc, sheet, meta).rows):
            location = row_location(sheet, meta, index, row)
            yield {'block_id': f"{sheet}!R{location['csv_record']}", 'text': row_text(row), 'location': location}


def search_documents(documents, query, *, limit=8):
    """Small-corpus lexical retrieval, restricted to the supplied selected documents."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query must be a nonempty string')
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError('limit must be 1..20')
    terms = set(re.findall(r'[\w]+', query.casefold())) - STOP
    hits = []
    for doc in documents.values():
        for unit in _units(doc):
            text = unit['text'].casefold()
            words = set(re.findall(r'[\w]+', text))
            score = len(terms & words)
            if query.casefold() in text:
                score += len(terms) + 1
            if score:
                hits.append({'document_id':doc.id, 'filename':doc.name, **unit, 'score':score})
    hits.sort(key=lambda hit: (-hit['score'], hit['filename'], hit['block_id']))
    return {'matches':hits[:limit], 'total_matching_blocks':len(hits),
            'truncated':len(hits)>limit,
            'retrieval_method':'lexical over text blocks and all spreadsheet rows; no semantic matching. '
                               'Use table tools for counts and totals.',
            'warnings':{doc.name:doc.warnings for doc in documents.values() if doc.warnings}}
