"""Compact metadata views for documents and workbook agents."""

QUALITY_KEYS = ('possible_variant_groups', 'possible_missing_markers', 'possible_placeholder_values',
                'all_values', 'frequent_values', 'min', 'max', 'negative_count')


def _column_hints(column):
    """Data-quality hints the planner needs before choosing exact filters."""
    hints = {key: column[key] for key in QUALITY_KEYS if key in column}
    # Value lists help with categories; long free-text values stay in get_sheet_schema only.
    if 'all_values' in hints and sum(len(v) for v in hints['all_values']) > 600:
        del hints['all_values']
    if 'possible_variant_groups' in hints:
        hints['possible_variant_groups'] = {g['suggested']: sorted(g['variants'])
                                            for g in hints['possible_variant_groups']}
    return hints


def sheet_index(metadata):
    index = {}
    for name, s in metadata.get('sheets', {}).items():
        columns = [c for c in s.get('columns', []) if not c.get('likely_junk')]
        entry = {'type': s.get('sheet_type'), 'rows': s.get('row_count'),
                 'columns': [c['name'] for c in columns]}
        hints = {c['name']: h for c in columns if (h := _column_hints(c))}
        if hints:
            entry['column_hints'] = hints
        index[name] = entry
    return index
