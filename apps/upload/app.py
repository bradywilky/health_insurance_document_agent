"""Document upload portal. Run: streamlit run apps/upload/app.py --server.port 8502"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from backend.preprocessing.documents import ingest_document, SUPPORTED
from backend.storage.s3 import configured_store

st.set_page_config(page_title='Document Upload Portal', layout='wide')
st.title('Document Upload Portal')
st.caption('Upload documents, extract their contents, and save them for the question-answering app.')
try:
    store = configured_store()
except Exception as exc:
    st.error(f'Cannot configure document storage: {exc}')
    st.stop()
st.caption('Storage: ' + ('local document library' if hasattr(store, 'directory') else 'S3 document library'))
st.info('Saving the same filename replaces the saved document. Use different filenames to keep both versions.')
uploaded = st.file_uploader('Upload documents', type=[s[1:] for s in sorted(SUPPORTED)],
                            accept_multiple_files=True)
st.caption('Up to 10 files per batch, 20 MB per file. Scanned PDFs require OCR; this portal reads text PDFs and Word body text/tables.')


def save_files(files):
    for name, content in files:
        try:
            with st.spinner(f'Processing {name}...'):
                doc = ingest_document(name, content, store=store)
            st.success(f'Saved {doc.name}')
            for warning in doc.warnings:
                st.warning(f'{doc.name}: {warning}')
        except Exception as exc:
            st.error(f'Could not save {name}: {type(exc).__name__}: {exc}')


if st.button('Process and save files', disabled=not uploaded):
    if len(uploaded) > 10:
        st.error('Select at most 10 files per batch.')
    else:
        save_files((file.name, file.getvalue()) for file in uploaded)

with st.expander('Development examples'):
    if st.button('Load Bingle-Dingle examples'):
        folder = Path(__file__).resolve().parents[2] / 'development' / 'fixtures' / 'documents'
        names = ['agreement', 'rates', 'amendment', 'processing_guide', 'draft',
                 'benefits', 'authorization', 'claim_packet']
        save_files((f'bingle_dingle_{name}.txt', (folder / f'bingle_dingle_{name}.txt').read_bytes())
                   for name in names)

st.caption('Open the chat app and click Refresh saved files to select your documents.')
