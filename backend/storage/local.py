"""Open a spreadsheet export written by `python -m backend.preprocessing.tables --output <dir>`."""
import json
from pathlib import Path

from backend.preprocessing.tables import SHEETS_FILE
from backend.storage.tables import MemoryTableFolder


def read_exported_tables(path):
    """(filename, metadata, folder) for an export directory: sheets.json plus each sheet's CSV."""
    root = Path(path).expanduser().resolve(strict=True)
    metadata = json.loads((root / SHEETS_FILE).read_text(encoding='utf-8'))
    files = {}
    for sheet in metadata['sheets'].values():
        name = sheet['csv_file']
        if '/' in name or '\\' in name or Path(name).name != name:
            raise ValueError('CSV file must be a basename inside the export directory')
        target = (root / name).resolve(strict=True)
        if target.parent != root:
            raise ValueError('CSV path escapes the export directory')
        files[name] = target.read_bytes()
    return metadata['filename'], metadata, MemoryTableFolder(files)


def exported_document(path):
    """A Document for an export directory, without reprocessing the original spreadsheet."""
    from hashlib import sha256
    from backend.preprocessing.documents import index_tables
    from backend.shared.models import Document
    filename, metadata, folder = read_exported_tables(path)
    fingerprint = json.dumps({'filename': filename, 'metadata': metadata}, sort_keys=True).encode()
    doc = Document(sha256(fingerprint).hexdigest()[:16], filename, Path(filename).suffix.lower().lstrip('.'),
                   table_folder=folder, table_metadata=metadata)
    index_tables(doc)
    return doc
