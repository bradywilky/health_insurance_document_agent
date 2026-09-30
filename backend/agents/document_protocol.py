"""Document tool schemas, evidence eligibility and source coverage."""

SOURCE = {'type':'string', 'pattern':'^(E[0-9]+|question)$'}


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
        spec('calculate','Exact decimal arithmetic. Write the expression with literal numbers copied from evidence, '
             'e.g. "2 * 88.00" or "(176.00 - 50.00) * 20 / 100", and list the evidence IDs (or "question") '
             'containing those numbers. Operators: + - * / ( ) min() max().',
             {'label':{'type':'string','minLength':1,'maxLength':200},
              'expression':{'type':'string','minLength':1,'maxLength':500},
              'sources':{'type':'array','items':SOURCE,'minItems':1,'maxItems':10,'uniqueItems':True}},
             ['label','expression','sources']),
        spec('date_calculate','Exact calendar arithmetic on ISO dates: add_days, days_between (other_date minus date) '
             'or compare (whether date is before/after other_date). Use for deadlines and effective-date checks. '
             'For "N days after X", date must be X itself (e.g. the restoration or service date), not another deadline.',
             {'label':{'type':'string','minLength':1,'maxLength':200},
              'operation':{'type':'string','enum':['add_days','days_between','compare']},
              'date':{'type':'string','format':'date'},'other_date':{'type':'string','format':'date'},
              'days':{'type':'integer','minimum':-3660,'maximum':3660},
              'sources':{'type':'array','items':SOURCE,'maxItems':10}},
             ['label','operation','date']),
        spec('answer','Finish research; a separate step writes the answer. Takes no arguments. Call alone.')]


def has_content(record):
    """True when a record holds source content (not an index or a derived calculation)."""
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


def examined_documents(evidence):
    """Selected document IDs from which source content was actually retrieved."""
    examined = set()
    for record in evidence:
        if not has_content(record):
            continue
        if record['tool'] == 'search_documents':
            examined.update(match['document_id'] for match in record['data']['matches'])
        else:
            examined.add(record['data']['document_id'])
    return examined
