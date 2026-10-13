"""Read a document's per-sheet CSVs from wherever its tables live.

A table folder is anything with read(csv_file) -> bytes. Saved documents use TableFolder (S3, or the local
document library through the same client interface); a file that was just preprocessed and not yet saved uses
MemoryTableFolder. Each sheet's metadata names its CSV in csv_file.
"""
from backend.shared.table import Table, parse_csv


def _check_name(name):
    if not isinstance(name, str) or not name or '/' in name or '\\' in name or name in {'.', '..'}:
        raise ValueError('Table file must be a basename inside the document folder')
    return name


class TableFolder:
    """A saved document's folder: <folder><csv_file> in `bucket`, read through an S3-style client."""

    def __init__(self, client, bucket, folder):
        if not folder.endswith('/'):
            raise ValueError('Table folder must end in /')
        self.client, self.bucket, self.folder = client, bucket, folder

    def read(self, name):
        with self.client.get_object(Bucket=self.bucket, Key=self.folder + _check_name(name))['Body'] as body:
            return body.read()


class MemoryTableFolder:
    """CSV bytes held in memory, e.g. a spreadsheet preprocessed in this process and not saved yet."""

    def __init__(self, files):
        self.files = {_check_name(name): body for name, body in files.items()}

    def read(self, name):
        if name not in self.files:
            raise FileNotFoundError(f'Table file not found: {name}')
        return self.files[name]


def load_table(folder, sheet_meta: dict) -> Table:
    """One sheet as a Table. Columns the metadata marks as text stay text (leading zeros, codes)."""
    text_columns = {c["name"] for c in sheet_meta.get("columns", [])
                    if c.get("dtype") in {"str", "string", "object"}}
    table = parse_csv(folder.read(sheet_meta["csv_file"]), text_columns)
    return table.select([c for c in table.columns if c.strip() != ""])
