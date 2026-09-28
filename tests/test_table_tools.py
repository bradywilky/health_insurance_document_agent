import pandas as pd
import pytest

from table_tools import query_table, join_tables
from evidence import numeric_display_check, bounded, compact


def test_filter_group_and_sort_decimal_precision():
    df = pd.DataFrame({'Group':['A','A','B','B'], 'Amount':[0.1,0.2,10,None],
                       'Status':['Approved']*4})
    result = query_table(df, {'filters':[{'column':'Status','op':'eq','value':'Approved'}],
        'group_by':['Group'], 'aggregations':[{'column':'Amount','op':'sum','as':'total'}],
        'sort':[{'column':'total','descending':True}]})
    assert result['rows'] == [{'Group':'B','total':'10.0'}, {'Group':'A','total':'0.3'}]
    assert result['null_counts']['Amount'] == 1


def test_all_missing_sum_is_not_zero():
    params = {'aggregations':[{'column':'x','op':'sum','as':'total'}]}
    assert query_table(pd.DataFrame({'x':[None,None]}), params)['rows'] == [{'total':None}]
    assert query_table(pd.DataFrame({'x':[None,0]}), params)['rows'] == [{'total':'0.0'}]


def test_counts_and_missing_group():
    df = pd.DataFrame({'group':['A',None,None], 'x':[1,2,None]})
    r = query_table(df, {'group_by':['group'], 'aggregations':[
        {'op':'count_rows','as':'rows'}, {'op':'count','column':'x','as':'populated'}]})
    assert r['rows'][1]['rows'] == 2
    assert r['rows'][1]['populated'] == 1


def test_literal_filter_and_pagination_preserve_source_indices():
    df = pd.DataFrame({'ID':['0012','0013','0014'], 'Name':['A.B','AxB','A.B']})
    result = query_table(df, {'filters':[{'column':'Name','op':'contains','value':'A.B'}],
                              'select':['ID'], 'limit':1, 'offset':1})
    assert result['rows'] == [{'ID':'0014'}]
    assert result['result_row_indices'] == [2]
    assert result['next_offset'] is None


@pytest.mark.parametrize('params', [
    {'sql':'DROP TABLE anything'}, {'select':['missing']}, {'limit':0}, {'offset':-1},
    {'group_by':['x']}, {'filters':[{'column':'x','op':'execute','value':'bad'}]},
    {'aggregations':[{'column':'x','op':'median','as':'m'}]},
    {'aggregations':[{'column':'x','op':'sum','as':'x'}], 'group_by':['x']},
])
def test_invalid_queries_fail_explicitly(params):
    with pytest.raises(ValueError):
        query_table(pd.DataFrame({'x':[1,2]}), params)


def test_non_numeric_sum_is_not_silently_coerced():
    with pytest.raises(ValueError, match='Non-numeric'):
        query_table(pd.DataFrame({'x':['12 oz','bad']}),
                    {'aggregations':[{'column':'x','op':'sum','as':'total'}]})


def test_join_reports_unmatched_and_never_matches_nulls():
    left = pd.DataFrame({'k':['a','b',None], 'amount':[1,2,3]})
    right = pd.DataFrame({'code':['a',None,'c'], 'label':['Alpha','Not a match','Gamma']})
    result = join_tables(left, right, {'left_on':['k'], 'right_on':['code']})
    assert result['diagnostics']['unmatched_left_rows'] == 2
    assert result['diagnostics']['unmatched_right_rows'] == 2
    assert result['diagnostics']['expected_output_rows'] == 3
    assert result['rows'][0]['label'] == 'Alpha'
    assert pd.isna(result['rows'][2]['label'])


def test_join_blocks_accidental_multiplication():
    left = pd.DataFrame({'k':['a','b'], 'amount':[10,20]})
    right = pd.DataFrame({'k':['a','a'], 'label':['One','Two']})
    result = join_tables(left, right, {'left_on':['k'],'right_on':['k']})
    assert 'error' in result
    assert result['diagnostics']['matched_rows_multiplied']
    assert result['diagnostics']['extra_matched_rows'] == 1
    assert result['diagnostics']['right_duplicate_key_rows'] == 2


def test_explicit_one_to_many_keeps_diagnostics_and_aggregate():
    left = pd.DataFrame({'k':['a','b'], 'amount':[10,20]})
    right = pd.DataFrame({'k':['a','a'], 'label':['One','Two']})
    result = join_tables(left, right, {'left_on':['k'],'right_on':['k'], 'relationship':'one_to_many',
        'query':{'aggregations':[{'column':'amount','op':'sum','as':'total'}]}})
    assert result['rows'] == [{'total':'40'}]
    assert result['diagnostics']['matched_rows_multiplied']


def test_join_cap_checked_before_materialization(monkeypatch):
    monkeypatch.setattr('table_tools.MAX_JOIN_ROWS', 3)
    result = join_tables(pd.DataFrame({'k':['a']*2}), pd.DataFrame({'k':['a']*2}),
                         {'left_on':['k'],'right_on':['k'],'relationship':'many_to_many'})
    assert 'exceeds' in result['error']


def test_join_filters_applied_before_cardinality_validation():
    result = join_tables(pd.DataFrame({'k':['a']}),
        pd.DataFrame({'k':['a','a'],'status':['Approved','Draft']}),
        {'left_on':['k'],'right_on':['k'], 'right_filters':[{'column':'status','op':'eq','value':'Approved'}]})
    assert 'error' not in result
    assert result['rows'][0]['status'] == 'Approved'


def test_numeric_display_keeps_exact_result():
    evidence = [{'id':'E1','data':{'aggregated':True,'decimal_columns':['total'],
                                 'rows':[{'total':'1860.776'}]}}]
    assert '1860.776' in numeric_display_check('Approximately 1861 g.', evidence)
    assert numeric_display_check('Total: 1,860.776 g.', evidence) == 'Total: 1,860.776 g.'


def test_bounded_result_remains_json_and_marks_omissions():
    data = {'rows':[{'text':'x'*3000}]*100}
    result = bounded(data, max_chars=2000)
    assert result['truncated'] is True
    assert len(compact(result)) <= 2000


def test_decimal_unit_conversion_before_aggregation():
    frame = pd.DataFrame({'amount':[16,500,2,None,0], 'unit':['oz','g','lb','oz','g']})
    result = query_table(frame, {'conversions':[{'column':'amount','unit_column':'unit',
        'factors':{'oz':'28.3495','g':'1','lb':'453.592'},'as':'grams'}],
        'aggregations':[{'column':'grams','op':'sum','as':'total_g'}]})
    assert result['rows'] == [{'total_g':'1860.77600'}]
    assert result['null_counts']['grams'] == 1


def test_missing_conversion_factor_cannot_silently_drop_records():
    with pytest.raises(ValueError,match='No conversion factor'):
        query_table(pd.DataFrame({'x':[1],'unit':['lb']}), {'conversions':[
            {'column':'x','unit_column':'unit','factors':{'oz':'28.3495'},'as':'g'}]})
