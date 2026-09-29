"""Saved-document selection and chat. Run: streamlit run apps/chat/app.py"""
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from backend.agents.document_agent import run_document_agent
from backend.agents.llm import LLMCallLog
from backend.storage.s3 import configured_store
from backend.config.settings import MODELS

st.set_page_config(page_title='Health Insurance Document Agent', layout='wide')
st.title('Health Insurance Document Agent')
st.caption('Choose your sources. Ask a question. Check the evidence.')
st.session_state.setdefault('messages', [])
st.session_state.setdefault('selection_signature', ())
try:
    store = configured_store(read_only=True)
except Exception as exc:
    st.error(f'Cannot configure document storage: {exc}')
    st.stop()


def clear_session():
    st.session_state.messages = []
    st.session_state.selected_keys = []
    st.session_state.documents = {}


with st.sidebar:
    st.header('Saved documents')
    st.caption('Add files through the separate upload portal, then refresh this list.')
    refresh = st.button('Refresh saved files')
    if refresh:
        st.session_state.messages = []
    try:
        saved = {item['manifest_key']: item for item in store.list_documents()}
    except Exception as exc:
        st.error(f'Could not list saved files: {exc}')
        st.stop()
    st.caption('Showing up to 100 saved documents. Select at most 10.')
    if 'selected_keys' in st.session_state:
        st.session_state.selected_keys = [k for k in st.session_state.selected_keys if k in saved]
    keys = st.multiselect('Files to answer from', list(saved), key='selected_keys',
        format_func=lambda key: saved[key]['filename'], max_selections=10,
        help='Only these files are available to the answering agent.')
    model = st.selectbox('Model', list(MODELS))
    if os.getenv('BEDROCK_MODEL_ID'):
        st.info('BEDROCK_MODEL_ID is set and overrides this model choice.')
    st.button('Clear selection and chat', on_click=clear_session)
    st.caption('Clearing the chat does not delete saved documents. Selected excerpts are sent to Amazon Bedrock.')

# Reload on every interaction, including questions, to detect replaced source files.
try:
    docs = [store.load(key) for key in keys]
except Exception as exc:
    st.session_state.documents = {}
    st.session_state.messages = []
    st.error(f'Could not open saved files: {exc}')
    st.stop()
st.session_state.documents = {doc.id: doc for doc in docs}
selected = list(st.session_state.documents)
signature = (tuple(sorted(selected)), model)
if signature != st.session_state.selection_signature:
    st.session_state.messages = []
    st.session_state.selection_signature = signature

if not selected:
    st.info('Select saved documents to ask a question. Add new documents in the upload portal.')
else:
    st.caption('Selected: ' + ', '.join(doc.name for doc in docs))
    with st.expander('Inspect extracted sources and warnings'):
        for doc in docs:
            st.subheader(doc.name)
            for warning in doc.warnings:
                st.warning(warning)
            st.json(doc.summary())
            st.json(doc.blocks[:3])


def show_evidence(result):
    protocol = result.get('protocol')
    if protocol:
        st.caption(f"{protocol['planner_mode']} tools · {protocol['repair_attempts']} corrective retries")
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
