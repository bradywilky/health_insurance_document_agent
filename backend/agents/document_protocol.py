"""Document tool schemas, evidence eligibility and source coverage."""

SOURCE = {'type':'string', 'pattern':'^(E[0-9]+|question)$'}


def tool_specs(document_ids, table_ids=None):
    """Native tool definitions; a single selected document gets no listing tool or per-search document filter.

    table_tool is offered only for table_ids (selected spreadsheets; None means every document) and is left out
    when there are none.
    """
    doc = {'type':'string', 'enum':list(document_ids)}
    table_ids = list(document_ids) if table_ids is None else list(table_ids)
    single = len(document_ids) == 1
    def spec(name, description, properties=None, required=None):
        return {'toolSpec':{'name':name, 'description':description, 'inputSchema':{'json':{
            'type':'object', 'properties':properties or {}, 'required':required or [],
            'additionalProperties':False}}}}
    search = {'query':{'type':'string','minLength':1,'maxLength':2000},
              'limit':{'type':'integer','minimum':1,'maximum':20}}
    if not single:
        search['document_ids'] = {'type':'array','items':doc,'minItems':1,'uniqueItems':True}
    scope = "the selected document's" if single else "selected documents'"
    listing = [] if single else [spec('list_documents','List selected documents; this is not content evidence.')]
    rows = ' and spreadsheet rows' if table_ids else ''
    tables = [spec('table_tool','Query an Excel/CSV using the table operations documented in the system prompt.',
                   {'document_id':{'type':'string','enum':table_ids}, 'name':{'type':'string','enum':[
                       'list_sheets','get_sheet_schema','read_sheet','search_all_sheets','query_table','join_tables']},
                    'parameters':{'type':'object'}}, ['document_id','name','parameters'])] if table_ids else []
    return listing + [
        spec('search',f'Search {scope} text{rows} lexically. Empty results do not prove absence.',
             search, ['query']),
        spec('read_document','Read consecutive source blocks. Follow next_offset for more content.',
             {'document_id':doc,'offset':{'type':'integer','minimum':0},
              'limit':{'type':'integer','minimum':1,'maximum':20}}, ['document_id'])] + tables + [
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
    if record['tool']=='search':
        return bool(data.get('matches'))
    if record['tool']=='table_tool':
        result = data.get('table_result',{})
        if result.get('error'):
            return False
        name = record['parameters']['name']
        if name=='search_all_sheets':
            # Hits are keyed by sheet name; a miss returns only a message and diagnostics.
            return any(key not in {'message','_search_details'} for key in result)
        # Sheet lists and schemas count: they describe this workbook's structure, computed from the file itself.
        return name in {'list_sheets','get_sheet_schema','read_sheet','query_table','join_tables'}
    return False


def examined_documents(evidence):
    """Selected document IDs from which source content was actually retrieved."""
    examined = set()
    for record in evidence:
        if not has_content(record):
            continue
        if record['tool'] == 'search':
            examined.update(match['document_id'] for match in record['data']['matches'])
        else:
            examined.add(record['data']['document_id'])
    return examined
