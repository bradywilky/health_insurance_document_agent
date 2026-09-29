"""Document tool schemas and evidence eligibility."""

def tool_specs(document_ids):
    doc = {'type':'string', 'enum':list(document_ids)}
    def spec(name, description, properties=None, required=None):
        return {'toolSpec':{'name':name, 'description':description, 'inputSchema':{'json':{
            'type':'object', 'properties':properties or {}, 'required':required or [],
            'additionalProperties':False}}}}
    return [
        spec('list_documents','List selected documents; this is not content evidence.'),
        spec('search_documents','Search selected documents lexically. Empty results do not prove absence.',
             {'query':{'type':'string','minLength':1,'maxLength':2000},
              'limit':{'type':'integer','minimum':1,'maximum':20},
              'document_ids':{'type':'array','items':doc,'minItems':1,'uniqueItems':True}}, ['query']),
        spec('read_document','Read consecutive source blocks. Follow next_offset for more content.',
             {'document_id':doc,'offset':{'type':'integer','minimum':0},
              'limit':{'type':'integer','minimum':1,'maximum':20}}, ['document_id']),
        spec('table_tool','Query an Excel/CSV using the table operations documented in the system prompt.',
             {'document_id':doc, 'name':{'type':'string','enum':[
                 'list_sheets','get_sheet_schema','read_sheet','search_all_sheets','query_table','join_tables']},
              'parameters':{'type':'object'}}, ['document_id','name','parameters']),
        spec('answer','Finish research. Does not accept answer text. Call alone after gathering source evidence.')]


def has_content(record):
    data = record['data']
    if data.get('error'):
        return False
    if record['tool']=='read_document':
        return bool(data.get('blocks'))
    if record['tool']=='search_documents':
        return bool(data.get('matches'))
    if record['tool']=='table_tool':
        return record['parameters']['name'] in {'read_sheet','query_table','join_tables'} and not data.get('table_result',{}).get('error')
    return False
