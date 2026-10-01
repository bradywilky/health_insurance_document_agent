"""Document ingestion and optional persistence for selected document Q&A."""
from backend.shared.models import Document
from backend.shared.rows import narrative_rows, row_location, row_text
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PAGES = 200
MAX_BLOCKS = 2000
TABLE_PREVIEW_ROWS = 200  # per data sheet; full rows stay searchable through the CSV
SUPPORTED = {'.pdf', '.docx', '.xlsx', '.xls', '.csv', '.txt', '.md'}


def _add_text(doc, text, location):
    text = text.replace('\x00', '').strip()
    for offset in range(0, len(text), 1400):
        if len(doc.blocks) >= MAX_BLOCKS:
            warning = f'Extraction limited to {MAX_BLOCKS} text blocks; some content is unavailable.'
            if warning not in doc.warnings:
                doc.warnings.append(warning)
            return
        # Overlap allows matches spanning a chunk boundary, without crossing source pages.
        doc.blocks.append({'block_id':f'B{len(doc.blocks)+1}', 'text':text[offset:offset+1600],
                           'location':location})


def ingest_document(name: str, content: bytes, *, store=None, enrich=None) -> Document:
    name = name.replace('\\','/').rsplit('/',1)[-1]
    extension = Path(name).suffix.lower()
    if extension not in SUPPORTED:
        raise ValueError(f'Unsupported file type: {extension}')
    if not content or len(content) > MAX_FILE_BYTES:
        raise ValueError('File must be nonempty and at most 20 MB')
    doc = Document(sha256(name.encode() + b'\0' + content).hexdigest()[:16], name, extension[1:])
    if extension == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(''):
            raise ValueError('Password-protected PDFs must be unlocked before upload')
        if len(reader.pages) > MAX_PAGES:
            doc.warnings.append(f'Only the first {MAX_PAGES} PDF pages were extracted.')
        for index, page in enumerate(reader.pages[:MAX_PAGES], 1):
            text = page.extract_text() or ''
            if not text.strip():
                doc.warnings.append(f'Page {index} has no extractable text; scanned pages require OCR.')
            else:
                _add_text(doc, text, {'page':index})
        doc.warnings.append('PDF text order and table layout may differ from the original; verify cited pages.')
    elif extension == '.docx':
        from docx import Document as WordDocument
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        word = WordDocument(BytesIO(content))
        paragraph_number, table_number, section = 0, 0, ''
        for element in word.element.body.iterchildren():
            if element.tag.endswith('}p'):
                paragraph = Paragraph(element, word)
                paragraph_number += 1
                if paragraph.style and paragraph.style.name.startswith('Heading'):
                    section = paragraph.text
                _add_text(doc, paragraph.text, {'paragraph':paragraph_number, 'section':section})
            elif element.tag.endswith('}tbl'):
                table_number += 1
                table = Table(element, word)
                for row_number, row in enumerate(table.rows, 1):
                    _add_text(doc, ' | '.join(cell.text for cell in row.cells),
                              {'table':table_number, 'row':row_number, 'section':section})
        doc.warnings.append('Word citations use paragraphs/tables, not page numbers. Headers, footers, text boxes and tracked changes are not fully extracted.')
    elif extension in {'.xlsx', '.xls', '.csv'}:
        from backend.storage.local import prepare_local_file
        with TemporaryDirectory(prefix='document-upload-') as directory:
            file = Path(directory) / name
            file.write_bytes(content)
            doc.table_inputs = prepare_local_file(file)
        _populate_table_document(doc)
    else:
        text = content.decode('utf-8-sig')
        for number, paragraph in enumerate(re.split(r'\n\s*\n', text), 1):
            _add_text(doc, paragraph, {'paragraph':number})
    if not doc.blocks:
        doc.warnings.append('No searchable text was extracted from this file.')
    if enrich is not None:
        # Optional LLM semantic profile (backend/agents/enrichment.py). Sends the extracted text to the model.
        try:
            doc.profile = enrich(doc)
        except Exception as exc:
            doc.warnings.append(f'Semantic profile not generated: {type(exc).__name__}: {exc}')
    if store is not None:
        store.save(doc, content)
    return doc


def _populate_table_document(doc):
    from backend.storage.tables import load_table
    inputs = doc.table_inputs
    key = f"{inputs['s3_prefix']}{inputs['plan_domain']}/preprocessed/{doc.name}/_metadata.json"
    with inputs['s3_client'].get_object(Bucket=inputs['s3_bucket'], Key=key)['Body'] as stream:
        doc.table_metadata = json.load(stream)
    for sheet, meta in doc.table_metadata['sheets'].items():
        frame = load_table(**doc.table_inputs, sheet_name=sheet, sheet_meta=meta)
        if meta.get('sheet_type') == 'metadata' and {'source_row', 'source_column', 'text'} <= set(frame.columns):
            # Narrative sheets are indexed in full, one block per sheet row (not per cell).
            for number, text in narrative_rows(frame):
                _add_text(doc, text, {'sheet': sheet, 'row': number})
            continue
        # Data sheets get a text preview; search_documents and table tools read every row from the CSV.
        preview = frame if meta.get('sheet_type') == 'metadata' else frame.head(TABLE_PREVIEW_ROWS)
        for i, row in preview.iterrows():
            _add_text(doc, row_text(row), row_location(sheet, meta, i, row))
        if len(preview) < len(frame):
            doc.warnings.append(f'{sheet}: text preview holds the first {len(preview)} of {len(frame)} rows. '
                                'search_documents and table tools read all rows.')
        for row in meta.get('context_rows', []):
            _add_text(doc, ' | '.join(row['values']), {'sheet':sheet, 'row':row['source_row']})
    doc.warnings.append('Spreadsheet ingestion uses heuristic headers; inspect ambiguous layouts. Formulas need cached values. Use table tools, not text snippets, for counts and totals.')


def document_from_table_inputs(*, s3_client, s3_bucket, s3_prefix, plan_domain, filename):
    """Load an existing local/S3 CSV-and-metadata export into the sole agent's document model.

    This reads preprocessing outputs; it does not invoke a model or save a new original.
    """
    inputs = dict(s3_client=s3_client, s3_bucket=s3_bucket, s3_prefix=s3_prefix,
                  plan_domain=plan_domain, filename=filename)
    doc = Document('', filename, Path(filename).suffix.lower().lstrip('.'), table_inputs=inputs)
    _populate_table_document(doc)
    fingerprint = json.dumps({'filename':filename, 'metadata':doc.table_metadata,
                              'blocks':doc.blocks}, sort_keys=True, default=str).encode()
    doc.id = sha256(fingerprint).hexdigest()[:16]
    return doc
