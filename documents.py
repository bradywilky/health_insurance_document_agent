"""Session-local ingestion for selected document Q&A. No shared storage or AWS writes."""
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PAGES = 200
MAX_BLOCKS = 2000
SUPPORTED = {'.pdf', '.docx', '.xlsx', '.xls', '.csv', '.txt', '.md'}


@dataclass
class Document:
    id: str
    name: str
    kind: str
    blocks: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    table_inputs: dict | None = None
    table_metadata: dict | None = None

    def summary(self):
        result = {'document_id':self.id, 'filename':self.name, 'type':self.kind,
                  'text_blocks':len(self.blocks), 'warnings':self.warnings}
        if self.table_metadata:
            from evidence import sheet_index
            result['sheets'] = sheet_index(self.table_metadata)
        return result


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


def ingest_document(name: str, content: bytes) -> Document:
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
        from local_data import prepare_local_file
        from table_agent import _load_df
        with TemporaryDirectory(prefix='document-upload-') as directory:
            file = Path(directory) / name
            file.write_bytes(content)
            doc.table_inputs = prepare_local_file(file)
        objects = doc.table_inputs['s3_client'].objects
        meta_key = next(k for k in objects if k.endswith('/_metadata.json'))
        doc.table_metadata = json.loads(objects[meta_key])
        for sheet, meta in doc.table_metadata['sheets'].items():
            frame = _load_df(**doc.table_inputs, sheet_name=sheet, sheet_meta=meta)
            for i, row in frame.iterrows():
                source = meta.get('source_rows', [])
                location = {'sheet':sheet, 'csv_record':int(i)+1}
                if meta.get('sheet_type') == 'metadata' and 'source_row' in row:
                    location['row'] = int(row['source_row'])
                elif i < len(source):
                    location['row'] = source[i]
                _add_text(doc, '; '.join(f'{col}: {value}' for col,value in row.items()), location)
            for row in meta.get('context_rows', []):
                _add_text(doc, ' | '.join(row['values']), {'sheet':sheet, 'row':row['source_row']})
        doc.warnings.append('Spreadsheet ingestion uses heuristic headers; inspect ambiguous layouts. Formulas need cached values. Text retrieval is not an exhaustive table calculation.')
    else:
        text = content.decode('utf-8-sig')
        for number, paragraph in enumerate(re.split(r'\n\s*\n', text), 1):
            _add_text(doc, paragraph, {'paragraph':number})
    if not doc.blocks:
        doc.warnings.append('No searchable text was extracted from this file.')
    return doc


def search_documents(documents, query, *, limit=8):
    """Small-corpus lexical retrieval, restricted to the supplied selected documents."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query must be a nonempty string')
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError('limit must be 1..20')
    stop = {'the','a','an','is','are','what','which','of','to','for','in','and','or','does','do','with'}
    terms = set(re.findall(r'[\w]+', query.casefold())) - stop
    hits = []
    for doc in documents.values():
        for block in doc.blocks:
            text = block['text'].casefold()
            words = set(re.findall(r'[\w]+', text))
            score = len(terms & words)
            if query.casefold() in text:
                score += len(terms) + 1
            if score:
                hits.append({'document_id':doc.id, 'filename':doc.name, **block, 'score':score})
    hits.sort(key=lambda hit: (-hit['score'], hit['filename'], hit['block_id']))
    return {'matches':hits[:limit], 'total_matching_blocks':len(hits),
            'truncated':len(hits)>limit,
            'retrieval_method':'lexical; no semantic matching or inferred policy precedence',
            'warnings':{doc.name:doc.warnings for doc in documents.values() if doc.warnings}}
