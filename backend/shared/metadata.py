"""Compact metadata views for documents and workbook backend.agents."""


def sheet_index(metadata):
    return {name: {'type': s.get('sheet_type'), 'rows': s.get('row_count'),
                   'columns': [c['name'] for c in s.get('columns', []) if not c.get('likely_junk')]}
            for name, s in metadata.get('sheets', {}).items()}
