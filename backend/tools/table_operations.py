"""Validated, deterministic table operations. No eval or generated Python."""
import operator
from collections import Counter
from decimal import Decimal, InvalidOperation, localcontext

from backend.shared.table import Table
from backend.shared.values import MISSING_MARKERS, placeholder_values, similar_values

MAX_JOIN_ROWS = 100_000
QUERY_KEYS = {'filters', 'select', 'group_by', 'aggregations', 'sort', 'offset', 'limit', 'conversions', 'recode',
              'derive'}
COMPARISONS = {'eq': operator.eq, 'ne': operator.ne, 'gt': operator.gt, 'ge': operator.ge, 'lt': operator.lt,
               'le': operator.le}


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


def _present(frame, col):
    return [v for v in frame.column(col) if v is not None]


def _value_counts(values):
    """Value -> count, most frequent first."""
    return dict(Counter(values).most_common())


def _coerce(frame, col, value):
    """Compare numeric columns numerically even when the value arrives as text ("-999999")."""
    if col in frame.numeric and isinstance(value, str):
        try:
            return float(value.replace(',', '').strip())
        except ValueError:
            raise ValueError(f'Column {col!r} is numeric; {value!r} is not a number') from None
    return value


def recode(frame, specs):
    """Apply explicit value mappings (variant -> canonical value, or -> null for missing markers).

    Unmapped values are unchanged. The mapping is part of the request, so it stays auditable.
    """
    if not isinstance(specs, list):
        raise ValueError('recode must be a list')
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
        values = frame.column(col)
        if col in frame.numeric:
            if any(v is not None for v in mapping.values()):
                raise ValueError('Numeric recodes may only map values to null')
            targets = [_coerce(frame, col, k) for k in mapping]
            frame = frame.with_column(col, [None if v in targets else v for v in values], numeric=True)
        else:
            unknown = [k for k in mapping if k not in set(values)]
            if unknown:
                raise ValueError(f'recode keys not present in {col!r}: {unknown[:10]}. Use exact values from the column.')
            recoded = [mapping.get(v, v) if v is not None else None for v in values]
            frame = frame.with_column(col, [None if v is None else str(v) for v in recoded])
    return frame


def derive(frame, specs):
    """Add columns holding one part of a text value, e.g. the prefix of "Address - Code" split on " - ".

    Lets questions group by a family, domain or prefix exactly instead of counting by eye.
    """
    if not isinstance(specs, list):
        raise ValueError('derive must be a list')
    for spec in specs:
        _check_keys(spec, {'column', 'as', 'split', 'part'})
        col, name, sep, part = spec.get('column'), spec.get('as'), spec.get('split'), spec.get('part', 0)
        _columns(frame, [col])
        if not isinstance(name, str) or not name or name in frame.columns:
            raise ValueError('derive "as" must name a new column')
        if not isinstance(sep, str) or not sep:
            raise ValueError('derive "split" must be a nonempty separator such as " - "')
        if type(part) is not int:
            raise ValueError('derive "part" must be an integer index (0 = first, -1 = last)')
        pieces = [None if v is None else str(v).split(sep) for v in frame.column(col)]
        frame = frame.with_column(name, [p[part].strip() if p is not None and -len(p) <= part < len(p) else None
                                         for p in pieces])
    return frame


def _matches(frame, col, op, value):
    """One boolean per row: does the row satisfy the condition? Missing values never match a comparison."""
    values = frame.column(col)
    if op in {'is_null', 'not_null'}:
        return [(v is None) == (op == 'is_null') for v in values]
    if op == 'contains':
        if not isinstance(value, str):
            raise ValueError('contains requires a string')
        needle = value.casefold()
        return [v is not None and needle in str(v).casefold() for v in values]
    if op == 'in':
        if not isinstance(value, list):
            raise ValueError('in requires a list')
        wanted = [_coerce(frame, col, v) for v in value]
        return [v is not None and v in wanted for v in values]
    if op in COMPARISONS:
        if value is None:
            raise ValueError('Use is_null/not_null to filter missing values')
        compare, value = COMPARISONS[op], _coerce(frame, col, value)
        try:
            return [v is not None and compare(v, value) for v in values]
        except TypeError:
            raise ValueError(f'Cannot compare values in {col!r} with {value!r}') from None
    raise ValueError(f'Unsupported filter operator: {op}')


def filter_rows(frame, filters):
    if not isinstance(filters, list):
        raise ValueError('filters must be a list')
    result = frame
    for condition in filters:
        _check_keys(condition, {'column', 'op', 'value'})
        col, op = condition['column'], condition['op']
        _columns(result, [col])
        mask = _matches(result, col, op, condition.get('value'))
        result = result.take(p for p, keep in enumerate(mask) if keep)
    return result


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
    present = _present(frame, col)
    if op == 'count':
        return len(present)
    if op == 'nunique':
        return len(set(present))
    if op not in {'sum', 'mean', 'min', 'max'}:
        raise ValueError(f'Unsupported aggregation: {op}')
    # Empty/all-missing sums must not masquerade as a genuine zero.
    if not present:
        return None
    numbers = [_decimal(value) for value in present]
    with localcontext() as ctx:
        ctx.prec = 38
        value = {'sum': lambda: sum(numbers), 'mean': lambda: sum(numbers) / len(numbers),
                 'min': lambda: min(numbers), 'max': lambda: max(numbers)}[op]()
    # Serialize decimals as strings to avoid introducing binary floating-point rounding.
    return format(value, 'f')


def _text_columns(frame):
    return [c for c in frame.columns if c not in frame.numeric and not str(c).startswith('_')]


def _found_elsewhere(frame, condition):
    """For a text filter that matches nothing, the other columns where its value does occur."""
    col, op, value = condition.get('column'), condition.get('op'), condition.get('value')
    wanted = [str(v).strip().casefold() for v in (value if isinstance(value, list) else [value])]
    found = {}
    for other in _text_columns(frame):
        if other == col:
            continue
        values = [str(v).strip().casefold() for v in _present(frame, other)]
        hits = sum(wanted[0] in v if op == 'contains' else v in wanted for v in values)
        if hits:
            found[other] = hits
    return found


def _filter_diagnostics(frame, filters):
    """Point out likely spelling variants, missing markers, or a value that sits in another column."""
    notes = []
    for condition in filters:
        col, op, value = condition.get('column'), condition.get('op'), condition.get('value')
        if col not in frame.columns or col in frame.numeric:
            continue
        if op in {'eq', 'in', 'contains'} and value is not None and not len(filter_rows(frame, [condition])):
            elsewhere = _found_elsewhere(frame, condition)
            if elsewhere:
                notes.append({'column': col, 'requested': value, 'matched_rows': 0, 'value_found_in_columns': elsewhere,
                              'message': 'This filter matched nothing in its column, but the value occurs in the '
                                         'columns listed. Check that the query uses the right column.'})
        counts = _value_counts(_present(frame, col))
        if len(counts) > 1000:
            continue
        counts = {str(k): n for k, n in counts.items()}
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


def _ordered(value):
    """Sort key that orders numbers before text, so a mixed column still sorts."""
    return (0, value, '') if isinstance(value, (int, float)) and not isinstance(value, bool) else (1, 0, str(value))


def _conflicting_rows(frame, filtered, params):
    """Rows that share an identifier-like key but disagree elsewhere (e.g. two different mappings).

    Category filters (status = "Partial") naturally return differing rows, so only keys whose values
    are mostly unique in the full table count.
    """
    def identifier_like(col):
        present = _present(frame, col)
        distinct = len(set(present))
        # Enough distinct values to judge, and mostly unique (a list of names or IDs, not a status column).
        return distinct >= 10 and distinct / len(present) >= 0.5
    keys = [c['column'] for c in params.get('filters', [])
            if c.get('op') in {'eq', 'in'} and c.get('column') in filtered.columns and identifier_like(c['column'])]
    shown = [c for c in (params.get('select') or filtered.columns)
             if c in filtered.columns and not str(c).startswith('_') and c not in keys]
    notes = []
    for key in keys[:1]:
        groups = {}
        for row in filtered.rows:
            if row[key] is not None:
                groups.setdefault(row[key], []).append(row)
        for value in sorted(groups, key=_ordered)[:5]:
            group = groups[value]
            if len(group) > 1 and shown:
                differing = [c for c in shown if len({None if r[c] is None else str(r[c]) for r in group}) > 1]
                if differing:
                    notes.append(f'{len(group)} rows have {key} = {value!r} but disagree in {differing}. '
                                 'Report every version with its sheet and row; do not pick one silently.')
    return notes


def _sort(frame, sort, decimal_columns):
    """Stable multi-column sort; missing values last for every column."""
    positions = list(range(len(frame)))
    for item in reversed(sort):
        col, descending = item['column'], item.get('descending', False)
        values = frame.column(col)
        key = (lambda v: Decimal(v)) if col in decimal_columns else (lambda v: v)
        present = [p for p in positions if values[p] is not None]
        try:
            present.sort(key=lambda p: key(values[p]), reverse=descending)
        except TypeError:
            raise ValueError(f'Column {col!r} mixes values that cannot be ordered') from None
        positions = present + [p for p in positions if values[p] is None]
    return frame.take(positions)


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
        if not isinstance(name, str) or not name or name in filtered.columns:
            raise ValueError('Conversion as must name a new column')
        factors = conversion['factors']
        if not isinstance(factors, dict) or not factors:
            raise ValueError('factors must map source unit labels to numeric multipliers')
        factors = {str(k): _decimal(v) for k, v in factors.items()}
        def convert(row):
            if row[column] is None:
                return None
            unit = str(row[unit_column])
            if unit not in factors:
                raise ValueError(f'No conversion factor for unit {unit!r}; no total was computed')
            with localcontext() as ctx:
                ctx.prec = 38
                return _decimal(row[column]) * factors[unit]
        filtered = filtered.with_column(name, [convert(row) for row in filtered.rows])
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
            _aggregate(filtered.take([]), spec)
        if group_by:
            # Groups in order of first appearance; a missing key is a group of its own.
            groups = {}
            for position, row in enumerate(filtered.rows):
                groups.setdefault(tuple(row[c] for c in group_by), []).append(position)
            groups = [(keys, filtered.take(positions)) for keys, positions in groups.items()]
        else:
            groups = [((), filtered)]
        rows = []
        for keys, group in groups:
            row = dict(zip(group_by, keys))
            row.update({spec['as']: _aggregate(group, spec) for spec in aggregates})
            rows.append(row)
        result = Table(group_by + aliases, rows, numeric=[c for c in group_by if c in filtered.numeric])
        numeric_columns = [s['as'] for s in aggregates if s['op'] in {'sum','mean','min','max'}]
    else:
        selected = params.get('select', list(filtered.columns))
        _columns(filtered, selected)
        result = filtered.select(selected)
    sort = params.get('sort', [])
    if not isinstance(sort, list):
        raise ValueError('sort must be a list')
    for item in sort:
        _check_keys(item, {'column', 'descending'})
        _columns(result, [item['column']])
        if not isinstance(item.get('descending', False), bool):
            raise ValueError('descending must be boolean')
    if sort:
        result = _sort(result, sort, numeric_columns)
    offset, limit = params.get('offset', 0), params.get('limit', 50)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('offset must be nonnegative and limit must be 1..100')
    page = result.take(range(offset, min(offset + limit, len(result))))
    notes = _conflicting_rows(frame, filtered, params) if not aggregates else []
    if len(result) > offset + len(page):
        notes.append(f'Only {len(page)} of {len(result)} result rows are shown. For distinct values use '
                     'group_by with count_rows; for totals use aggregations; or page with offset.')
    placeholders = {}
    for spec in aggregates:
        col = spec.get('column')
        if spec.get('op') in {'sum', 'mean', 'min', 'max'} and col in frame.columns and col in filtered.columns:
            suspicious = set(placeholder_values(frame.column(col)))
            found = {str(k): n for k, n in _value_counts(_present(filtered, col)).items() if str(k) in suspicious}
            if found:
                placeholders[col] = found
    if placeholders:
        notes.append(f'Aggregated values include possible placeholder numbers {placeholders}; consider '
                     'reporting results with and without them (filter ne, or recode to null).')
    # Carry available unit labels alongside numeric projections, even when the model omits them.
    unit_columns = [c for c in filtered.columns if str(c).lower() in {'unit','units','source unit'}]
    unit_context = ({c: [str(v) for v in list(dict.fromkeys(_present(filtered, c)))[:20]] for c in unit_columns}
                    if not aggregates else {})
    return {'rows': page.records(), 'matched_rows': len(filtered),
            'total_result_rows': len(result), 'next_offset': offset + limit if offset + limit < len(result) else None,
            'null_counts': {str(c): sum(v is None for v in filtered.column(c)) for c in filtered.columns},
            'decimal_columns': numeric_columns, 'aggregated': bool(aggregates),
            'result_row_indices': [] if aggregates else list(page.index),
            'contributing_row_indices': filtered.index[:100],
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
    for a, b in zip(left_on, right_on):
        if (a in left.numeric) != (b in right.numeric):
            raise ValueError(f'Cannot join text column and numeric column on keys {a!r} and {b!r}')
    reserved = {'__left_index','__right_index'}
    if reserved & (set(left.columns) | set(right.columns)):
        raise ValueError('Source columns collide with reserved join provenance columns')
    def keys(frame, cols):
        return [tuple(row[c] for c in cols) for row in frame.rows]
    left_valid = left.take(p for p, key in enumerate(keys(left, left_on)) if None not in key)
    right_valid = right.take(p for p, key in enumerate(keys(right, right_on)) if None not in key)
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
    # A key with the same name on both sides appears once; other shared names get _left/_right suffixes.
    shared = {a for a, b in zip(left_on, right_on) if a == b}
    overlap = (set(left.columns) & set(right.columns)) - shared
    left_names = {c: c + '_left' if c in overlap else c for c in left.columns}
    right_names = {c: c + '_right' if c in overlap else c for c in right.columns if c not in shared}
    partners = {}
    for position, key in enumerate(keys(right_valid, right_on)):
        partners.setdefault(key, []).append(position)
    # Null keys never match: they were left out of right_valid, and a left key holding None finds no partner.
    rows, provenance = [], []
    for position, (row, key) in enumerate(zip(left.rows, keys(left, left_on))):
        out = {left_names[c]: row[c] for c in left.columns}
        matches = partners.get(key, []) if None not in key else []
        if not matches and how == 'left':
            rows.append({**out, **dict.fromkeys(right_names.values())})
            provenance.append({'__left_index': left.index[position], '__right_index': None})
        for match in matches:
            other = right_valid.rows[match]
            rows.append({**out, **{name: other[c] for c, name in right_names.items()}})
            provenance.append({'__left_index': left.index[position], '__right_index': right_valid.index[match]})
    joined = Table(list(left_names.values()) + list(right_names.values()), rows,
                   numeric=[left_names[c] for c in left.numeric] + [right_names[c] for c in right.numeric
                                                                     if c in right_names])
    result = query_table(joined, params.get('query', {}))
    indices = result['contributing_row_indices'] if result['aggregated'] else result['result_row_indices']
    result['join_provenance'] = [provenance[i] for i in indices]
    result['diagnostics'] = diagnostics
    return result
