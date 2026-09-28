import json
from pathlib import Path

import pandas as pd
import pytest

from local_data import prepare_local_file, prepare_preprocessed_directory
from preprocess import normalize_sheet, preprocess_file
from table_agent import _execute_tool




@pytest.fixture
def workbook(insurance_workbook):
    return prepare_local_file(insurance_workbook), preprocess_file(insurance_workbook)[1]


def tool(workbook, name, **parameters):
    inputs, meta = workbook
    return _execute_tool(name, parameters, **inputs, metadata=meta)


def test_messy_headers_and_provenance(workbook):
    sheets = workbook[1]['sheets']
    cast = sheets['Meadow Mapping']
    assert cast['header_row'] == 4
    assert cast['row_count'] == 6
    assert cast['source_rows'] == [5, 6, 7, 9, 11, 12]
    assert any('Repeated header' in w for w in cast['warnings'])
    assert cast['context_rows'][-1]['source_row'] == 13
    assert sheets['River Mapping']['row_count'] == 7
    assert any('Duplicate records retained: 1' in w for w in sheets['River Mapping']['warnings'])
    names = [c['name'] for c in sheets['Provider Mapping']['columns']]
    assert 'Notes__2' in names


def test_metadata_late_rows_and_pagination(workbook):
    assert workbook[1]['sheets']['About']['sheet_type'] == 'metadata'
    assert workbook[1]['sheets']['Version History']['sheet_type'] == 'metadata'
    page = tool(workbook, 'read_sheet', sheet_name='About', offset=5, limit=5)
    assert page['next_offset'] is None
    assert page['rows'][-1]['source_row'] == 10
    assert 'USD' in page['rows'][-1]['text']
    assert 'About' in tool(workbook, 'search_all_sheets', term='USD')


def test_text_codes_zero_and_missing(workbook):
    page = tool(workbook, 'read_sheet', sheet_name='Claim Samples')
    rows = page['rows']
    assert rows[0]['Claim ID'] == '0012'
    assert rows[3]['Service Code'] == 'NA'
    assert rows[3]['Billed Amount'] is None
    assert rows[4]['Billed Amount'] == 0
    assert 'Claim Samples' in tool(workbook, 'search_all_sheets', term='0012')
    assert 'Service Codes' in tool(workbook, 'search_all_sheets', term='NA')


def test_ground_truth_billed_total(workbook):
    rows = tool(workbook, 'read_sheet', sheet_name='Claim Samples')['rows']
    assert sum(r['Billed Amount'] for r in rows if r['Billed Amount'] is not None) == 445


def test_profiles_and_export_contract(workbook, insurance_workbook):
    objects, meta = preprocess_file(insurance_workbook)
    assert len(objects) == 9
    assert '_metadata.json' in objects
    assert json.loads(objects['_metadata.json']) == meta
    mass = next(c for c in meta['sheets']['Claim Samples']['columns'] if c['name'] == 'Billed Amount')
    assert mass['null_count'] == 1
    assert mass['distinct_count'] == 4


def test_explicit_override_and_unknown_name(insurance_workbook):
    _, meta = preprocess_file(insurance_workbook, overrides={'Version History': {'sheet_type':'data', 'header_row':1}})
    assert meta['sheets']['Version History']['row_count'] == 3
    with pytest.raises(ValueError, match='Override names'):
        preprocess_file(insurance_workbook, overrides={'typo': {}})


def test_blank_header_column_and_duplicate_names():
    raw = pd.DataFrame([['ID','Notes','Notes',None], ['0012','a','b',None]])
    frame, meta = normalize_sheet('Mappings', raw)
    assert list(frame.columns) == ['ID','Notes','Notes__2','Unnamed: 4']
    assert meta['columns'][-1]['likely_junk'] is True


def test_ragged_csv_keeps_identifiers(tmp_path):
    file = tmp_path / 'ragged.csv'
    file.write_text('Insurance export\n\nID,Value\n0012,10\n0013,0\n', encoding='utf-8')
    objects, meta = preprocess_file(file)
    assert meta['sheets']['Sheet1']['header_row'] == 3
    assert b'0012' in objects['ragged.csv']


def test_exported_directory_round_trip(tmp_path, insurance_workbook):
    objects, meta = preprocess_file(insurance_workbook)
    for name, body in objects.items():
        (tmp_path / name).write_bytes(body)
    inputs = prepare_preprocessed_directory(tmp_path)
    rows = _execute_tool('read_sheet', {'sheet_name':'Claim Samples'}, **inputs, metadata=meta)['rows']
    assert rows[0]['Claim ID'] == '0012'


def test_exported_directory_rejects_traversal(tmp_path):
    meta = {'filename':'book.xlsx', 'sheets':{'Bad': {'csv_file':'../outside.csv'}}}
    (tmp_path / '_metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='basename'):
        prepare_preprocessed_directory(tmp_path)
