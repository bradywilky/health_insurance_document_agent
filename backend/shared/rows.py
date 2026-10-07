"""One text form and source location for a table row, shared by preprocessing and search."""
import re

MIDNIGHT = re.compile(r"^(\d{4}-\d{2}-\d{2}) 00:00:00$")


def row_text(row):
    return '; '.join(f'{col}: {"" if value is None else value}' for col, value in row.items())


def narrative_rows(cells):
    """Rebuild narrative-sheet cells (dicts with source_row, source_column, text) into one text line per sheet row.

    A multi-cell row whose columns repeat in the rows below acts as a header, so small embedded
    tables (revision logs, sign-offs) read as "Date: ...; Version: ...; Author: ...".
    Yields (source_row, text).
    """
    rows = {}
    for cell in cells:
        # Spreadsheet dates arrive as midnight timestamps; show the date alone.
        text = MIDNIGHT.sub(r'\1', str(cell['text']))
        rows.setdefault(int(cell['source_row']), {})[int(cell['source_column'])] = text
    ordered = sorted(rows.items())
    header = None
    for position, (number, cells) in enumerate(ordered):
        columns = sorted(cells)
        following = ordered[position + 1][1] if position + 1 < len(ordered) else {}
        if len(cells) < 2:
            header = None
            yield number, cells[columns[0]]
        elif header and set(cells) <= set(header):
            yield number, '; '.join(f'{header[c]}: {cells[c]}' for c in columns)
        else:
            # A new header only when the next row uses the same columns; otherwise plain text.
            header = cells if following and set(following) <= set(cells) else None
            yield number, ' | '.join(cells[c] for c in columns)


def row_location(sheet, meta, index, row):
    location = {'sheet': sheet, 'csv_record': int(index) + 1}
    source = meta.get('source_rows', [])
    if meta.get('sheet_type') == 'metadata' and 'source_row' in row:
        location['row'] = int(row['source_row'])
    elif index < len(source):
        location['row'] = source[index]
    return location
