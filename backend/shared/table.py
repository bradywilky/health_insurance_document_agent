"""A small in-memory table read from a preprocessed CSV, using only the standard library.

Rows are dicts keyed by column name; None is a missing value. Each row keeps its position in the
source CSV (`index`) through filtering, sorting and selection, so results can cite the rows they came from.
"""
import csv
import io
import re

NUMBER = re.compile(r'\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*')
INTEGER = re.compile(r'\s*[+-]?\d+\s*')


class Table:
    def __init__(self, columns, rows, index=None, numeric=()):
        self.columns = list(columns)
        self.rows = rows
        self.index = list(range(len(rows))) if index is None else list(index)
        self.numeric = {c for c in numeric if c in self.columns}

    def __len__(self):
        return len(self.rows)

    def column(self, name):
        return [row[name] for row in self.rows]

    def take(self, positions):
        """The rows at these positions (not source indices), keeping their source indices."""
        positions = list(positions)
        return Table(self.columns, [self.rows[p] for p in positions], [self.index[p] for p in positions],
                     self.numeric)

    def select(self, columns):
        return Table(columns, [{c: row[c] for c in columns} for row in self.rows], self.index, self.numeric)

    def with_column(self, name, values, numeric=False):
        """A copy with column `name` added or replaced."""
        columns = self.columns if name in self.columns else self.columns + [name]
        kinds = (self.numeric | {name}) if numeric else (self.numeric - {name})
        return Table(columns, [{**row, name: value} for row, value in zip(self.rows, values)], self.index, kinds)

    def records(self):
        return [{c: row[c] for c in self.columns} for row in self.rows]


def is_numeric(values):
    """True when every present value is a number (an all-missing column counts as numeric, as in pandas)."""
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values if v is not None)


def from_records(records, columns=None):
    """A table from row dicts; columns holding only numbers are numeric."""
    columns = list(columns or dict.fromkeys(k for r in records for k in r))
    rows = [{c: r.get(c) for c in columns} for r in records]
    return Table(columns, rows, numeric=[c for c in columns if is_numeric(r[c] for r in rows)])


def concat(tables):
    """Stack tables; a column missing from a table is None there, and stays numeric only if numeric everywhere."""
    columns = list(dict.fromkeys(c for t in tables for c in t.columns))
    rows = [{c: row.get(c) for c in columns} for t in tables for row in t.rows]
    numeric = [c for c in columns if all(c in t.numeric for t in tables if c in t.columns)]
    return Table(columns, rows, numeric=numeric)


def _header(names):
    """Name blank headers "Unnamed: i" and number repeats "Name.1", as pandas.read_csv does."""
    out, seen = [], set()
    for i, name in enumerate(names):
        name = name if name != '' else f'Unnamed: {i}'
        base, n = name, 0
        while name in seen:
            n += 1
            name = f'{base}.{n}'
        seen.add(name)
        out.append(name)
    return out


def parse_csv(raw, text_columns=()):
    """Parse CSV bytes. Empty cells are missing. A column is numeric when every present value is a number,
    unless it is named in text_columns (so codes such as "0012" keep their zeros). A numeric column holding
    any decimal is all floats; otherwise all integers.
    """
    lines = [line for line in csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline='')) if line]
    if not lines:
        return Table([], [])
    columns = _header(lines[0])
    width = len(columns)
    rows = [{c: (v if v != '' else None) for c, v in zip(columns, line[:width] + [''] * (width - len(line)))}
            for line in lines[1:]]
    numeric = []
    for col in columns:
        present = [row[col] for row in rows if row[col] is not None]
        if col in text_columns or not all(NUMBER.fullmatch(v) for v in present):
            continue
        convert = int if all(INTEGER.fullmatch(v) for v in present) else float
        for row in rows:
            if row[col] is not None:
                row[col] = convert(row[col])
        numeric.append(col)
    return Table(columns, rows, numeric=numeric)
