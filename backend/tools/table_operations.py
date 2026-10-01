"""Validated, deterministic table operations. No eval or generated Python."""
from decimal import Decimal, InvalidOperation, localcontext

import pandas as pd

from backend.preprocessing.profiling import MISSING_MARKERS, placeholder_values, similar_values

MAX_JOIN_ROWS = 100_000
QUERY_KEYS = {'filters', 'select', 'group_by', 'aggregations', 'sort', 'offset', 'limit', 'conversions', 'recode',
              'derive'}


def _check_keys(params, allowed):
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f'Unsupported parameters: {sorted(unknown)}')


def _columns(frame, names):
    if not isinstance(names, list) or not all(isinstance(c, str) for c in names):
        raise ValueError('Column names must be a list of strings')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate selected columns are not allowed')
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f'Unknown columns: {sorted(missing)}; available: {list(frame.columns)}')


def _coerce(series, value):
    """Compare numeric columns numerically even when the value arrives as text ("-999999")."""
    if (pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            and isinstance(value, str)):
        try:
            return float(value.replace(',', '').strip())
        except ValueError:
            raise ValueError(f'Column {series.name!r} is numeric; {value!r} is not a number') from None
    return value


def recode(frame, specs):
    """Apply explicit value mappings (variant -> canonical value, or -> null for missing markers).

    Unmapped values are unchanged. The mapping is part of the request, so it stays auditable.
    """
    if not isinstance(specs, list):
        raise ValueError('recode must be a list')
    frame = frame.copy()
    for spec in specs:
        _check_keys(spec, {'column', 'map'})
        col, mapping = spec['column'], spec['map']
        _columns(frame, [col])
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('recode map must be a nonempty object of {"original": "replacement or null"}')
        if all(isinstance(v, list) for v in mapping.values()):
            # Also accept the grouped form {"canonical": ["variant", ...]}; it is unambiguous.
            inverted = {}
            for target, variants in mapping.items():
                for variant in variants:
                    if not isinstance(variant, str) or variant in inverted:
                        raise ValueError('Grouped recode lists must hold distinct strings')
                    inverted[variant] = target
            mapping = inverted
        elif not all(v is None or isinstance(v, str) for v in mapping.values()):
            raise ValueError('recode map values must be strings or null, e.g. {"THR": "Tehran", "nan": null}')
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series):
            if any(v is not None for v in mapping.values()):
                raise ValueError('Numeric recodes may only map values to null')
            targets = {_coerce(series, k) for k in mapping}
            frame[col] = series.where(~series.isin(targets))
        else:
            unknown = [k for k in mapping if k not in set(series.dropna())]
            if unknown:
                raise ValueError(f'recode keys not present in {col!r}: {unknown[:10]}. Use exact values from the column.')
            frame[col] = series.map(lambda v: mapping.get(v, v) if pd.notna(v) else v).astype('string')
    return frame


def derive(frame, specs):
    """Add columns holding one part of a text value, e.g. the prefix of "Address - Code" split on " - ".

    Lets questions group by a family, domain or prefix exactly instead of counting by eye.
    """
    if not isinstance(specs, list):
        raise ValueError('derive must be a list')
    frame = frame.copy()
    for spec in specs:
        _check_keys(spec, {'column', 'as', 'split', 'part'})
        col, name, sep, part = spec.get('column'), spec.get('as'), spec.get('split'), spec.get('part', 0)
        _columns(frame, [col])
        if not isinstance(name, str) or not name or name in frame:
            raise ValueError('derive "as" must name a new column')
        if not isinstance(sep, str) or not sep:
            raise ValueError('derive "split" must be a nonempty separator such as " - "')
        if type(part) is not int:
            raise ValueError('derive "part" must be an integer index (0 = first, -1 = last)')
        pieces = frame[col].astype('string').str.split(sep, regex=False)
        frame[name] = pieces.map(lambda p: p[part].strip() if isinstance(p, list) and -len(p) <= part < len(p)
                                 else pd.NA).astype('string')
    return frame


def filter_rows(frame, filters):
    if not isinstance(filters, list):
        raise ValueError('filters must be a list')
    result = frame
    for condition in filters:
        _check_keys(condition, {'column', 'op', 'value'})
        col, op = condition['column'], condition['op']
        _columns(result, [col])
        series, value = result[col], condition.get('value')
        if op in {'is_null', 'not_null'}:
            mask = series.isna() if op == 'is_null' else series.notna()
        elif op == 'contains':
            if not isinstance(value, str):
                raise ValueError('contains requires a string')
            mask = series.astype('string').str.contains(value, case=False, regex=False, na=False)
        elif op == 'in':
            if not isinstance(value, list):
                raise ValueError('in requires a list')
            mask = series.isin([_coerce(series, v) for v in value])
        elif op in {'eq', 'ne', 'gt', 'ge', 'lt', 'le'}:
            if value is None:
                raise ValueError('Use is_null/not_null to filter missing values')
            mask = getattr(series, op)(_coerce(series, value)) & series.notna()
        else:
            raise ValueError(f'Unsupported filter operator: {op}')
        result = result.loc[mask.fillna(False)]
    return result.copy()


def _decimal(value):
    if isinstance(value, bool):
        raise ValueError('Boolean values are not numeric measurements')
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f'Non-numeric value cannot be aggregated: {value!r}') from exc
    if not number.is_finite():
        raise ValueError('Non-finite numeric values are not supported')
    return number


def _aggregate(frame, spec):
    _check_keys(spec, {'op', 'column', 'as'})
    op = spec['op']
    if op == 'count_rows':
        return len(frame)
    col = spec.get('column')
    _columns(frame, [col])
    present = frame[col].dropna()
    if op == 'count':
        return len(present)
    if op == 'nunique':
        return int(present.nunique())
    if op not in {'sum', 'mean', 'min', 'max'}:
        raise ValueError(f'Unsupported aggregation: {op}')
    # Empty/all-missing sums must not masquerade as a genuine zero.
    if present.empty:
        return None
    numbers = [_decimal(value) for value in present]
    with localcontext() as ctx:
        ctx.prec = 38
        value = {'sum': lambda: sum(numbers), 'mean': lambda: sum(numbers) / len(numbers),
                 'min': lambda: min(numbers), 'max': lambda: max(numbers)}[op]()
    # Serialize decimals as strings to avoid introducing binary floating-point rounding.
    return format(value, 'f')


def _text_columns(frame):
    return [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c]) and not str(c).startswith('_')]


def _found_elsewhere(frame, condition):
    """For a text filter that matches nothing, the other columns where its value does occur."""
    col, op, value = condition.get('column'), condition.get('op'), condition.get('value')
    wanted = [str(v).strip().casefold() for v in (value if isinstance(value, list) else [value])]
    found = {}
    for other in _text_columns(frame):
        if other == col:
            continue
        values = frame[other].dropna().astype('string').str.strip().str.casefold()
        mask = (values.str.contains(wanted[0], regex=False) if op == 'contains' else values.isin(wanted))
        if mask.any():
            found[other] = int(mask.sum())
    return found


def _filter_diagnostics(frame, filters):
    """Point out likely spelling variants, missing markers, or a value that sits in another column."""
    notes = []
    for condition in filters:
        col, op, value = condition.get('column'), condition.get('op'), condition.get('value')
        if col not in frame or pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if op in {'eq', 'in', 'contains'} and value is not None and filter_rows(frame, [condition]).empty:
            elsewhere = _found_elsewhere(frame, condition)
            if elsewhere:
                notes.append({'column': col, 'requested': value, 'matched_rows': 0, 'value_found_in_columns': elsewhere,
                              'message': 'This filter matched nothing in its column, but the value occurs in the '
                                         'columns listed. Check that the query uses the right column.'})
        counts = frame[col].dropna().value_counts()
        if len(counts) > 1000:
            continue
        counts = {str(k): int(n) for k, n in counts.items()}
        if op in {'eq', 'in'}:
            requested = value if isinstance(value, list) else [value]
            others = similar_values([str(v) for v in requested], counts)
            if others:
                notes.append({'column': col, 'requested': requested, 'similar_values_not_matched': others,
                              'message': 'Exact matching excluded these similar spellings. If they are the same '
                                         'entity, use recode or an "in" list, and say so in the answer.'})
        elif op == 'is_null':
            markers = {k: n for k, n in counts.items() if k.strip().casefold() in MISSING_MARKERS}
            if markers:
                notes.append({'column': col, 'possible_missing_markers': markers,
                              'message': 'These text values may mean missing but are not null. Include them '
                                         'with recode (value -> null) or an "in" filter if appropriate.'})
    return notes


def _conflicting_rows(frame, filtered, params):
    """Rows that share an identifier-like key but disagree elsewhere (e.g. two different mappings).

    Category filters (status = "Partial") naturally return differing rows, so only keys whose values
    are mostly unique in the full table count.
    """
    def identifier_like(col):
        present = frame[col].dropna()
        # Enough distinct values to judge, and mostly unique (a list of names or IDs, not a status column).
        return present.nunique() >= 10 and present.nunique() / len(present) >= 0.5
    keys = [c['column'] for c in params.get('filters', [])
            if c.get('op') in {'eq', 'in'} and c.get('column') in filtered and identifier_like(c['column'])]
    shown = [c for c in (params.get('select') or list(filtered.columns))
             if c in filtered and not str(c).startswith('_') and c not in keys]
    notes = []
    for key in keys[:1]:
        for value, group in list(filtered.groupby(key, dropna=True))[:5]:
            if len(group) > 1 and shown:
                differing = [c for c in shown if group[c].astype('string').nunique(dropna=False) > 1]
                if differing:
                    notes.append(f'{len(group)} rows have {key} = {value!r} but disagree in {differing}. '
                                 'Report every version with its sheet and row; do not pick one silently.')
    return notes


def query_table(frame, params):
    _check_keys(params, QUERY_KEYS)
    frame = derive(recode(frame, params.get('recode', [])), params.get('derive', []))
    filtered = filter_rows(frame, params.get('filters', []))
    conversions = params.get('conversions', [])
    if not isinstance(conversions, list):
        raise ValueError('conversions must be a list')
    for conversion in conversions:
        _check_keys(conversion, {'column', 'unit_column', 'factors', 'as'})
        column, unit_column, name = conversion['column'], conversion['unit_column'], conversion['as']
        _columns(filtered, [column, unit_column])
        if not isinstance(name, str) or not name or name in filtered:
            raise ValueError('Conversion as must name a new column')
        factors = conversion['factors']
        if not isinstance(factors, dict) or not factors:
            raise ValueError('factors must map source unit labels to numeric multipliers')
        factors = {str(k): _decimal(v) for k, v in factors.items()}
        def convert(row):
            if pd.isna(row[column]):
                return None
            unit = str(row[unit_column])
            if unit not in factors:
                raise ValueError(f'No conversion factor for unit {unit!r}; no total was computed')
            with localcontext() as ctx:
                ctx.prec = 38
                return _decimal(row[column]) * factors[unit]
        filtered[name] = pd.Series([convert(row) for _, row in filtered.iterrows()],
                                   index=filtered.index, dtype=object)
    group_by = params.get('group_by', [])
    _columns(filtered, group_by)
    aggregates = params.get('aggregations', [])
    if not isinstance(aggregates, list):
        raise ValueError('aggregations must be a list')
    if group_by and not aggregates:
        raise ValueError('group_by requires aggregations')
    numeric_columns = []
    if aggregates:
        if params.get('select'):
            raise ValueError('select cannot be combined with aggregations')
        aliases = [spec.get('as') for spec in aggregates]
        if any(not isinstance(alias, str) or not alias for alias in aliases):
            raise ValueError('Each aggregation requires a nonempty as name')
        if len(set(aliases)) != len(aliases) or set(aliases) & set(group_by):
            raise ValueError('Aggregation output names must be unique and distinct from group keys')
        # Validate even if there are no matching groups.
        for spec in aggregates:
            _aggregate(filtered.iloc[:0], spec)
        groups = filtered.groupby(group_by, dropna=False, sort=False) if group_by else [((), filtered)]
        rows = []
        for keys, group in groups:
            keys = keys if isinstance(keys, tuple) else (keys,)
            row = dict(zip(group_by, keys))
            row.update({spec['as']: _aggregate(group, spec) for spec in aggregates})
            rows.append(row)
        result = pd.DataFrame(rows, columns=group_by + aliases)
        numeric_columns = [s['as'] for s in aggregates if s['op'] in {'sum','mean','min','max'}]
    else:
        selected = params.get('select', list(filtered.columns))
        _columns(filtered, selected)
        result = filtered[selected]
    sort = params.get('sort', [])
    if not isinstance(sort, list):
        raise ValueError('sort must be a list')
    for item in sort:
        _check_keys(item, {'column', 'descending'})
        _columns(result, [item['column']])
        if not isinstance(item.get('descending', False), bool):
            raise ValueError('descending must be boolean')
    if sort:
        result = result.sort_values([s['column'] for s in sort],
                                    ascending=[not s.get('descending', False) for s in sort],
                                    kind='stable', na_position='last',
                                    key=lambda s: s.map(lambda v: Decimal(v) if pd.notna(v) else None)
                                    if s.name in numeric_columns else s)
    offset, limit = params.get('offset', 0), params.get('limit', 50)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('offset must be nonnegative and limit must be 1..100')
    page = result.iloc[offset:offset + limit]
    notes = _conflicting_rows(frame, filtered, params) if not aggregates else []
    if len(result) > offset + len(page):
        notes.append(f'Only {len(page)} of {len(result)} result rows are shown. For distinct values use '
                     'group_by with count_rows; for totals use aggregations; or page with offset.')
    placeholders = {}
    for spec in aggregates:
        col = spec.get('column')
        if spec.get('op') in {'sum', 'mean', 'min', 'max'} and col in frame and col in filtered:
            suspicious = set(placeholder_values(frame[col]))
            found = {str(k): int(n) for k, n in filtered[col].value_counts().items() if str(k) in suspicious}
            if found:
                placeholders[col] = found
    if placeholders:
        notes.append(f'Aggregated values include possible placeholder numbers {placeholders}; consider '
                     'reporting results with and without them (filter ne, or recode to null).')
    # Carry available unit labels alongside numeric projections, even when the model omits them.
    unit_columns = [c for c in filtered if str(c).lower() in {'unit','units','source unit'}]
    unit_context = ({c: [str(v) for v in filtered[c].dropna().unique()[:20]] for c in unit_columns}
                    if not aggregates else {})
    return {'rows': page.to_dict(orient='records'), 'matched_rows': len(filtered),
            'total_result_rows': len(result), 'next_offset': offset + limit if offset + limit < len(result) else None,
            'null_counts': {str(c): int(filtered[c].isna().sum()) for c in filtered.columns},
            'decimal_columns': numeric_columns, 'aggregated': bool(aggregates),
            'result_row_indices': [] if aggregates else [int(i) for i in page.index],
            'contributing_row_indices': [int(i) for i in filtered.index[:100]],
            'provenance_truncated': len(filtered) > 100,
            'unit_context': unit_context, 'conversions': conversions,
            'numeric_policy': 'Decimal arithmetic, 38 significant digits; means may repeat. No implicit rounding.',
            'recode': params.get('recode', []), 'derive': params.get('derive', []),
            'filter_diagnostics': _filter_diagnostics(frame, params.get('filters', [])), 'notes': notes}


def join_tables(left, right, params):
    _check_keys(params, {'left_on','right_on','how','relationship','left_filters','right_filters','query'})
    left = filter_rows(left, params.get('left_filters', []))
    right = filter_rows(right, params.get('right_filters', []))
    left_on, right_on = params['left_on'], params['right_on']
    _columns(left, left_on)
    _columns(right, right_on)
    if not left_on or len(left_on) != len(right_on):
        raise ValueError('Join keys must be nonempty lists of equal length')
    how = params.get('how', 'left')
    relationship = params.get('relationship', 'many_to_one')
    if how not in {'left','inner'}:
        raise ValueError('how must be left or inner')
    if relationship not in {'one_to_one','many_to_one','one_to_many','many_to_many'}:
        raise ValueError('Invalid relationship')
    reserved = {'__left_index','__right_index'}
    if reserved & (set(left.columns) | set(right.columns)):
        raise ValueError('Source columns collide with reserved join provenance columns')
    left_valid = left[left[left_on].notna().all(axis=1)]
    right_valid = right[right[right_on].notna().all(axis=1)]
    def keys(frame, cols):
        return list(frame[cols].itertuples(index=False, name=None))
    from collections import Counter
    lc, rc = Counter(keys(left_valid, left_on)), Counter(keys(right_valid, right_on))
    matched_left = sum(n for key, n in lc.items() if key in rc)
    matched_right = sum(n for key, n in rc.items() if key in lc)
    matched_output = sum(n * rc.get(key, 0) for key, n in lc.items())
    unmatched_left = len(left) - matched_left
    expected_rows = matched_output + (unmatched_left if how == 'left' else 0)
    diagnostics = {'left_rows': len(left), 'right_rows': len(right),
                   'left_duplicate_key_rows': sum(n for n in lc.values() if n > 1),
                   'right_duplicate_key_rows': sum(n for n in rc.values() if n > 1),
                   'left_null_key_rows': len(left)-len(left_valid),
                   'right_null_key_rows': len(right)-len(right_valid),
                   'unmatched_left_rows': unmatched_left, 'unmatched_right_rows': len(right)-matched_right,
                   'expected_output_rows': expected_rows,
                   'matched_rows_multiplied': matched_output > matched_left,
                   'extra_matched_rows': matched_output - matched_left,
                   'relationship': relationship, 'null_policy': 'Null keys never match'}
    invalid = ((relationship in {'one_to_one','one_to_many'} and any(n > 1 for n in lc.values()))
               or (relationship in {'one_to_one','many_to_one'} and any(n > 1 for n in rc.values())))
    if invalid:
        return {'error': 'Join cardinality does not match relationship; resolve duplicates or explicitly choose the intended relationship.',
                'diagnostics': diagnostics}
    if expected_rows > MAX_JOIN_ROWS:
        return {'error': f'Join exceeds {MAX_JOIN_ROWS} output rows; filter first.', 'diagnostics': diagnostics}
    left = left.assign(__left_index=left.index)
    right_valid = right_valid.assign(__right_index=right_valid.index)
    # Removing null keys from the right prevents pandas' null-equals-null join behavior.
    joined = left.merge(right_valid, left_on=left_on, right_on=right_on, how=how,
                        suffixes=('_left','_right'), sort=False)
    provenance = joined[['__left_index','__right_index']].copy()
    result = query_table(joined.drop(columns=['__left_index','__right_index']), params.get('query', {}))
    indices = result['contributing_row_indices'] if result['aggregated'] else result['result_row_indices']
    result['join_provenance'] = provenance.iloc[indices].where(provenance.notna(), None).to_dict(orient='records')
    result['diagnostics'] = diagnostics
    return result
