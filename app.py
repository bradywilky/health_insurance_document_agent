"""Local multi-file research prototype. Run: streamlit run app.py"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from documents import ingest_document, SUPPORTED
from health_insurance_document_agent import run_document_agent
from llm import LLMCallLog

st.set_page_config(page_title='Health Insurance Document Agent', layout='wide')
st.title('Health Insurance Document Agent')
st.caption('Choose your sources. Ask a question. Check the evidence.')
st.session_state.setdefault('documents', {})
st.session_state.setdefault('messages', [])
st.session_state.setdefault('selection_signature', ())


def clear_session():
    for key in ['documents','messages','selection_signature','selected_ids','uploads']:
        st.session_state.pop(key,None)


def load_examples():
    folder = Path(__file__).resolve().parent / 'fixtures' / 'documents'
    names = ['bingle_dingle_agreement.txt', 'bingle_dingle_rates.txt',
              'bingle_dingle_amendment.txt', 'bingle_dingle_processing_guide.txt',
              'bingle_dingle_draft.txt', 'bingle_dingle_benefits.txt',
              'bingle_dingle_authorization.txt', 'bingle_dingle_claim_packet.txt']
    docs = [ingest_document(name, (folder / name).read_bytes()) for name in names]
    st.session_state.documents = {doc.id: doc for doc in docs}
    st.session_state.selected_ids = list(st.session_state.documents)
    st.session_state.messages = []

with st.sidebar:
    st.header('Your files')
    st.caption('Local prototype. Files stay in this app session; selected excerpts are sent to Amazon Bedrock.')
    uploaded = st.file_uploader('Upload documents', type=[s[1:] for s in sorted(SUPPORTED)],
                                accept_multiple_files=True, key='uploads')
    if st.button('Add uploaded files', disabled=not uploaded):
        if len(uploaded) + len(st.session_state.documents) > 10:
            st.error('This prototype supports at most 10 files per session. Clear the session to start a new set.')
        else:
            for file in uploaded:
                try:
                    with st.spinner(f'Reading {file.name}…'):
                        doc = ingest_document(file.name,file.getvalue())
                    st.session_state.documents[doc.id] = doc
                except Exception as exc:
                    st.error(f'Could not read {file.name}: {type(exc).__name__}: {exc}')
    ids = list(st.session_state.documents)
    selected = st.multiselect('Files to answer from', ids,
        format_func=lambda key:f'{st.session_state.documents[key].name} · {key[:6]}',
        key='selected_ids', help='Only these files are available to the answering agent.')
    model = st.selectbox('Model', ['maverick','scout'])
    if os.getenv('BEDROCK_MODEL_ID'):
        st.info('BEDROCK_MODEL_ID is set and overrides this model choice.')
    st.button('Clear files and chat', on_click=clear_session)
    st.button('Load Bingle-Dingle examples', on_click=load_examples,
              help='Replaces the current file set and chat with eight Bingle-Dingle health-insurance documents.')
    st.caption('PDF scans need OCR. This version reads text-based PDFs and Word body text/tables.')

signature = tuple(sorted(selected))
if signature != st.session_state.selection_signature:
    st.session_state.messages = []
    st.session_state.selection_signature = signature

if not selected:
    st.info('Upload files, click “Add uploaded files”, then select the files you want to ask about.')
else:
    st.caption('Selected: ' + ', '.join(st.session_state.documents[k].name for k in selected))
    with st.expander('Inspect extracted sources and warnings'):
        for key in selected:
            doc = st.session_state.documents[key]
            st.subheader(doc.name)
            for warning in doc.warnings:
                st.warning(warning)
            st.json(doc.summary())
            st.json(doc.blocks[:3])


def show_evidence(result):
    for warning in result.get('limitations',[]):
        st.warning(warning)
    with st.expander('Evidence used'):
        for record in result.get('evidence',[]):
            data = record['data']
            st.markdown(f"**{record['id']}**")
            passages = data.get('matches',data.get('blocks',[]))
            for passage in passages:
                location = ', '.join(f'{k} {v}' for k,v in passage.get('location',{}).items() if v != '')
                st.caption(f"{passage.get('filename',data.get('filename',''))} — {location}")
                st.text(passage['text'])
            if 'table_result' in data:
                table = data['table_result']
                st.caption(data.get('filename',''))
                if table.get('rows'):
                    st.dataframe(table['rows'], hide_index=True)
                else:
                    st.json(table)
                if table.get('diagnostics'):
                    st.json(table['diagnostics'])
            if data.get('error'):
                st.warning(data['error'])
            if data.get('truncated') or data.get('next_offset') is not None:
                st.caption('This evidence contains only part of the available text.')


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if 'result' in message:
            show_evidence(message['result'])

question = st.chat_input('Ask about the selected files', disabled=not selected)
if question:
    history = [{'role':m['role'],'content':m['content']} for m in st.session_state.messages]
    st.session_state.messages.append({'role':'user','content':question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        status = st.empty()
        log = LLMCallLog()
        try:
            # A per-call model setting is passed to the agent; never share user file caches.
            result = run_document_agent(st.session_state.documents,selected,question,
                history=history,call_log=log,
                on_step=lambda _:status.caption('Checking the selected sources…'),model=model)
            status.empty()
            st.markdown(result['answer'])
            show_evidence(result)
            st.caption(f"{len(log.records)} model calls · "
                       f"{sum(r['usage'].get('inputTokens',0) for r in log.records):,} input tokens")
            st.session_state.messages.append({'role':'assistant','content':result['answer'],'result':result})
        except Exception as exc:
            status.empty()
            if 'expired' in str(exc).lower() or 'LoginRefreshRequired' in type(exc).__name__:
                st.error('AWS login has expired. Renew your configured AWS profile, then try again.')
            else:
                st.error(f'The question could not be completed: {type(exc).__name__}: {exc}')
