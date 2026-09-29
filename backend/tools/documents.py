"""Selected-document tools, independent of agent orchestration."""
from backend.retrieval.search import search_documents
from backend.tools.tables import execute_table_tool

def selected_documents(documents, ids):
    if not ids:
        raise ValueError('Select at least one document')
    missing = set(ids) - set(documents)
    if missing:
        raise ValueError('One or more selected documents are unavailable')
    return {key:documents[key] for key in dict.fromkeys(ids)}


def _document(docs, params):
    key = params.get('document_id')
    if key not in docs:
        raise ValueError('Document is not in the selected file set')
    return docs[key]


def execute_document_tool(name, params, docs):
    if name == 'list_documents':
        return {'documents':[doc.summary() for doc in docs.values()]}
    if name == 'search_documents':
        subset = selected_documents(docs, params['document_ids']) if 'document_ids' in params else docs
        return search_documents(subset, params['query'], limit=params.get('limit',8))
    if name == 'read_document':
        doc = _document(docs, params)
        offset, limit = params.get('offset',0), params.get('limit',8)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError('offset must be nonnegative and limit must be 1..20')
        return {'document_id':doc.id, 'filename':doc.name, 'blocks':doc.blocks[offset:offset+limit],
                'next_offset':offset+limit if offset+limit<len(doc.blocks) else None,
                'total_blocks':len(doc.blocks), 'warnings':doc.warnings}
    if name == 'table_tool':
        doc = _document(docs, params)
        allowed = {'list_sheets','get_sheet_schema','read_sheet','search_all_sheets','query_table','join_tables'}
        if params.get('name') not in allowed:
            raise ValueError('Unsupported table tool; generated code execution is disabled')
        if doc.table_inputs is None:
            raise ValueError('This file is not an Excel or CSV table')
        return {'document_id':doc.id,'filename':doc.name,
                'table_result':execute_table_tool(params['name'],params.get('parameters',{}),
                                              **doc.table_inputs, metadata=doc.table_metadata)}
    raise ValueError(f'Unknown document tool: {name}')
