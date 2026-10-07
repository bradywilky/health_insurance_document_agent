"""Messy categorical data: variant spellings, text missing markers, placeholder numbers, large sheets."""
import json

import pandas as pd
import pytest

from backend.agents.document_agent import run_document_agent
from backend.preprocessing.documents import TABLE_PREVIEW_ROWS, ingest_document
from backend.preprocessing.profiling import column_quality, variant_groups
from backend.retrieval.search import search_documents
from backend.tools.documents import execute_document_tool
from backend.tools.table_operations import query_table
from table_helpers import as_table

ROWS = [('fail', 'Visa', 'Denver', -999999), ('FAIL', 'VISA', 'DENVER', 120.0), ('failed', 'Vsa', 'DEN', 80.0),
        ('success', 'MasterCard', 'Denver ', 50.0), ('Success', 'Master Card', 'Boulder', 0.0),
        ('success', 'nan', 'nan', 30.0), ('success', 'Amex', 'boulder', -999999)]


@pytest.fixture
def frame():
    d = pd.DataFrame(ROWS * 3, columns=['status', 'card', 'city', 'amount'])
    for col in ['status', 'card', 'city']:
        d[col] = d[col].astype('string')
    return d


def count(frame, **params):
    return query_table(as_table(frame), {**params, 'aggregations': [{'op': 'count_rows', 'as': 'n'}]})


def test_profile_suggests_variants_markers_and_placeholders(frame):
    city = column_quality(frame['city'])
    groups = {g['suggested']: set(g['variants']) for g in city['possible_variant_groups']}
    assert groups == {'Denver': {'Denver', 'DENVER', 'DEN', 'Denver '}, 'Boulder': {'Boulder', 'boulder'}}
    assert city['possible_missing_markers'] == {'nan': 3}
    assert 'nan' not in str(city['possible_variant_groups'])
    amount = column_quality(frame['amount'])
    assert amount['possible_placeholder_values'] == {'-999999.0': 6}
    assert amount['negative_count'] == 6


@pytest.mark.parametrize('values', [
    ['Medicare', 'Medicaid', 'Medical', 'Medicaid A'], ['Male', 'Female'], ['Dental', 'Rental'],
    ['Plan A', 'Plan B'], ['Tier 1', 'Tier 2', 'Tier 10'], ['Premium', 'Premium Plus'],
    ['In-network', 'Out-of-network'], ['Inpatient', 'Outpatient'], ['NA', 'NY', 'NJ'],
    ['DEMO-SESSION', 'DEMO-VISIT', 'DEMO-REMOTE'], ['Approved', 'Draft', 'Denied'],
])
def test_distinct_values_are_never_grouped(values):
    assert variant_groups({v: 2 for v in values}) == []


def test_literal_na_code_is_not_a_missing_marker():
    assert 'possible_missing_markers' not in column_quality(pd.Series(['NA', 'DEMO', 'NA'], dtype='string'))


def test_numeric_filters_accept_text_numbers(frame):
    assert count(frame, filters=[{'column': 'amount', 'op': 'eq', 'value': '-999999'}])['rows'] == [{'n': 6}]
    assert count(frame, filters=[{'column': 'amount', 'op': 'in', 'value': ['0', 50]}])['rows'] == [{'n': 6}]
    with pytest.raises(ValueError, match='numeric'):
        count(frame, filters=[{'column': 'amount', 'op': 'eq', 'value': 'lots'}])


def test_exact_filter_reports_unmatched_spellings(frame):
    result = count(frame, filters=[{'column': 'city', 'op': 'eq', 'value': 'Denver'}])
    assert result['rows'] == [{'n': 3}]
    diag = result['filter_diagnostics'][0]
    assert diag['similar_values_not_matched'] == {'DENVER': 3, 'DEN': 3, 'Denver ': 3}


def test_is_null_reports_text_missing_markers(frame):
    result = count(frame, filters=[{'column': 'card', 'op': 'is_null'}])
    assert result['rows'] == [{'n': 0}]
    assert result['filter_diagnostics'][0]['possible_missing_markers'] == {'nan': 3}


@pytest.mark.parametrize('mapping', [
    {'DENVER': 'Denver', 'DEN': 'Denver', 'Denver ': 'Denver'},
    {'Denver': ['Denver', 'DENVER', 'DEN', 'Denver ']},
])
def test_recode_combines_variants_in_either_form(frame, mapping):
    result = count(frame, recode=[{'column': 'city', 'map': mapping}],
                   filters=[{'column': 'city', 'op': 'eq', 'value': 'Denver'}])
    assert result['rows'] == [{'n': 12}] and result['filter_diagnostics'] == []
    grouped = query_table(as_table(frame), {'recode': [{'column': 'city', 'map': mapping}], 'group_by': ['city'],
                                  'aggregations': [{'op': 'count_rows', 'as': 'n'}]})
    assert {r['city']: r['n'] for r in grouped['rows']}['Denver'] == 12


def test_recode_to_null_and_validation(frame):
    assert count(frame, recode=[{'column': 'card', 'map': {'nan': None}}],
                 filters=[{'column': 'card', 'op': 'is_null'}])['rows'] == [{'n': 3}]
    total = query_table(as_table(frame), {'recode': [{'column': 'amount', 'map': {'-999999': None}}],
                                'aggregations': [{'column': 'amount', 'op': 'sum', 'as': 't'}]})
    assert total['rows'] == [{'t': '840.0'}]
    with pytest.raises(ValueError, match='not present'):
        count(frame, recode=[{'column': 'city', 'map': {'Denvr': 'Denver'}}])
    with pytest.raises(ValueError, match='strings or null'):
        count(frame, recode=[{'column': 'city', 'map': {'DEN': 5}}])
    with pytest.raises(ValueError, match='only map values to null'):
        count(frame, recode=[{'column': 'amount', 'map': {'0': 'zero'}}])


def test_notes_flag_truncated_lists_and_placeholder_totals(frame):
    listed = query_table(as_table(frame), {'select': ['city'], 'limit': 5})
    assert 'Only 5 of 21 result rows' in listed['notes'][0]
    total = query_table(as_table(frame), {'aggregations': [{'column': 'amount', 'op': 'sum', 'as': 't'}]})
    assert '-999999.0' in total['notes'][0]


def large_csv(rows=TABLE_PREVIEW_ROWS + 300):
    lines = ['Code,City,Amount'] + [f'C{i:04},{"Tehran" if i % 2 else "THR"},{i}' for i in range(rows)]
    lines[-1] = f'LASTROW,Boulder,{rows}'
    return ingest_document('large.csv', '\n'.join(lines).encode())


def test_large_sheet_preview_is_labeled_and_search_reads_every_row():
    doc = large_csv()
    assert len(doc.blocks) == TABLE_PREVIEW_ROWS
    assert any('first 200 of 500 rows' in w for w in doc.warnings)
    assert not any('Extraction limited' in w for w in doc.warnings)
    hit = search_documents({doc.id: doc}, 'LASTROW')['matches'][0]
    assert hit['location']['csv_record'] == 500 and hit['block_id'] == 'Sheet1!R500'
    # Preview rows are not double counted: each row is one searchable unit.
    assert search_documents({doc.id: doc}, 'THR')['total_matching_blocks'] == 250


def test_planner_summary_carries_column_hints():
    doc = large_csv()
    hints = doc.summary()['sheets']['Sheet1']['column_hints']
    # THR 250 vs Tehran 249 (the last row became Boulder), so the most frequent spelling leads.
    assert hints['City']['possible_variant_groups'] == {'THR': ['THR', 'Tehran']}
    assert hints['City']['all_values'] == {'THR': 250, 'Tehran': 249, 'Boulder': 1}


def test_schema_alone_can_support_a_data_quality_answer(native_script):
    doc = large_csv()
    native_script([
        {'tool': 'table_tool', 'parameters': {'document_id': doc.id, 'name': 'get_sheet_schema',
                                              'parameters': {'sheet_name': 'Sheet1'}}},
        {'tool': 'answer', 'parameters': {}}, 'City has variant spellings THR and Tehran [E1].'])
    result = run_document_agent({doc.id: doc}, [doc.id], 'Any data quality issues?')
    assert result['protocol']['repair_attempts'] == 0
    assert result['answer'].startswith('City has variant')


def test_recode_through_the_document_tool():
    doc = large_csv()
    result = execute_document_tool('table_tool', {'document_id': doc.id, 'name': 'query_table', 'parameters': {
        'sheet_name': 'Sheet1', 'recode': [{'column': 'City', 'map': {'THR': 'Tehran'}}],
        'filters': [{'column': 'City', 'op': 'eq', 'value': 'Tehran'}],
        'aggregations': [{'op': 'count_rows', 'as': 'n'}]}}, {doc.id: doc})['table_result']
    assert result['rows'] == [{'n': 499}]
    assert json.dumps(result['recode']) == '[{"column": "City", "map": {"THR": "Tehran"}}]'


def test_tie_prefers_readable_spelling_over_code():
    groups = variant_groups({'THR': 5, 'Tehran': 5, 'Tehran ': 5})
    assert groups[0]['suggested'] == 'Tehran'
