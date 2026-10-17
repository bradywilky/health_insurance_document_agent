"""Document upload portal. Run: streamlit run apps/upload/app.py --server.port 8502"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from backend.preprocessing.documents import ingest_document, SUPPORTED
from backend.storage.s3 import configured_store
from backend.agents.enrichment import make_enricher
from backend.agents.search_hints import make_hinter
from backend.agents.embeddings import make_embedder
from backend.config.settings import semantic_search_enabled

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
profile_at_upload = st.checkbox(
    'Generate a semantic profile (document type, status, dates, references, glossary)',
    help='Sends each file\'s full extracted text to Amazon Bedrock once at upload. The profile guides the '
         'answering agent but is never cited as evidence. Review it below.')
hints_at_upload = st.checkbox(
    'Generate search hints (other words for each passage)',
    help='Sends each file\'s extracted text to Amazon Bedrock once at upload, about one call per 15 passages. '
         'Keyword search then also finds passages worded differently from the question. Hints are never evidence.')
# Shown only with SEMANTIC_SEARCH=true; environments without a Bedrock embedding model leave it off.
embed_at_upload = semantic_search_enabled() and st.checkbox(
    'Embed passages for semantic search', value=True,
    help='Sends each passage to the Bedrock embedding model once at upload (one call per passage). The answering '
         'agent can then find passages by meaning, not only by shared words.')


def save_files(files):
    for name, content in files:
        try:
            # ingest_document traces and archives the upload, including failures.
            with st.spinner(f'Processing {name}...'):
                enrich = make_enricher() if profile_at_upload else None
                hint = make_hinter() if hints_at_upload else None
                embed = make_embedder() if embed_at_upload else None
                doc = ingest_document(name, content, store=store, enrich=enrich, hint=hint, embed=embed)
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

st.header('Review semantic profiles')
st.caption('Confirm or correct what the model inferred. Confirmed glossary meanings are marked for the answering '
           'agent; inferred ones stay labeled as guesses.')
try:
    saved = {item['manifest_key']: item for item in store.list_documents()}
except Exception as exc:
    st.error(f'Could not list saved files: {exc}')
    saved = {}
key = st.selectbox('Saved document', list(saved), index=None, format_func=lambda k: saved[k]['filename'],
                   placeholder='Choose a saved document')
if key:
    doc = store.load(key)
    profile = doc.profile
    label = 'Regenerate profile' if profile else 'Generate profile'
    if st.button(label, help='Sends the extracted text to Amazon Bedrock. Regenerating discards earlier review.'):
        with st.spinner('Generating profile...'):
            try:
                generated = make_enricher()(doc)
                store.save_profile(key, generated)
                st.rerun()
            except Exception as exc:
                st.error(f'Could not generate a profile: {type(exc).__name__}: {exc}')
    hints = (doc.search_hints or {}).get('meta')
    st.caption(f"Search hints: {hints['blocks_with_hints']} of {hints['blocks_total']} passages" if hints
               else 'No search hints for this document yet.')
    if doc.blocks and st.button('Regenerate search hints' if hints else 'Generate search hints',
                                help='Sends the extracted text to Amazon Bedrock.'):
        with st.spinner('Generating search hints...'):
            try:
                store.save_search_hints(key, make_hinter()(doc))
                st.rerun()
            except Exception as exc:
                st.error(f'Could not generate search hints: {type(exc).__name__}: {exc}')
    if semantic_search_enabled():
        embedded = (doc.embeddings or {}).get('meta')
        st.caption(f"Embeddings: {embedded['blocks_embedded']} of {embedded['blocks_total']} passages "
                   f"({embedded['model_id']})" if embedded else 'No embeddings for this document yet.')
        if doc.blocks and st.button('Regenerate embeddings' if embedded else 'Generate embeddings',
                                    help='Sends each passage to the Bedrock embedding model.'):
            with st.spinner('Embedding passages...'):
                try:
                    store.save_embeddings(key, make_embedder()(doc))
                    st.rerun()
                except Exception as exc:
                    st.error(f'Could not generate embeddings: {type(exc).__name__}: {exc}')
    if not profile:
        st.info('No semantic profile for this document yet.')
    else:
        meta = profile.get('meta', {})
        st.markdown(f"**{profile.get('document_type') or 'Unknown type'}** · status: {profile.get('status', 'unknown')}"
                    + (' · reviewed' if meta.get('reviewed') else ''))
        st.write(profile.get('summary', ''))
        for title, field, show in [
                ('Effective dates', 'effective_dates', lambda i: f"{i['label']}: {i['value']}"),
                ('Identifiers', 'identifiers', lambda i: i['identifier']),
                ('References to other documents', 'references', lambda i: f"{i['relationship']} {i['target']}"),
                ('Status values', 'status_values', lambda i: f"{i['value']}: {i['meaning']}")]:
            if profile.get(field):
                st.markdown(f'**{title}**')
                st.markdown('\n'.join(f'- {show(i)} — "{i.get("quote", "")}"' for i in profile[field]))
        st.markdown('**Glossary**')
        edited = st.data_editor(
            [{'term': g['term'], 'meaning': g['meaning'], 'source': g['source'], 'quote': g.get('quote', '')}
             for g in profile.get('glossary', [])],
            num_rows='dynamic', key=f'glossary-{key}', use_container_width=True,
            column_config={'source': st.column_config.SelectboxColumn(options=['defined', 'inferred', 'confirmed'],
                                                                      required=True),
                           'quote': st.column_config.TextColumn(disabled=True)})
        if st.button('Save reviewed profile'):
            rows = edited.to_dict('records') if hasattr(edited, 'to_dict') else list(edited)
            locations = {g['term']: g.get('location') for g in profile.get('glossary', [])}
            glossary = [{'term': r['term'].strip(), 'meaning': (r.get('meaning') or '').strip(),
                         'source': r.get('source') or 'inferred', 'quote': r.get('quote') or '',
                         'location': locations.get(r['term'])} for r in rows if (r.get('term') or '').strip()]
            from datetime import datetime, timezone
            reviewed = {**profile, 'glossary': glossary,
                        'meta': {**meta, 'reviewed': True,
                                 'reviewed_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}}
            try:
                store.save_profile(key, reviewed)
                st.success('Saved. The chat app uses the reviewed profile after Refresh saved files.')
            except Exception as exc:
                st.error(f'Could not save: {type(exc).__name__}: {exc}')
        dropped = {k: v for k, v in profile.get('dropped', {}).items() if v}
        if dropped:
            st.caption(f'Claims removed or downgraded because their quotes were not found in the document: {dropped}')
        if not meta.get('coverage', {}).get('complete', True):
            st.warning('This profile covers only the beginning of a long document.')
