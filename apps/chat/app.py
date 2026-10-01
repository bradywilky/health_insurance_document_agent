"""Saved-document selection and chat. Run: streamlit run apps/chat/app.py"""
import os
from dotenv import load_dotenv
load_dotenv()
import uuid
import streamlit as st
from backend.observability.tracing import configure_tracing
from backend.services.questions import answer_question
from backend.storage.s3 import configured_store
from backend.config.settings import MODELS

AMBIGUITY_MODES = {'Off: answer directly': 'off', 'State assumptions': 'assumptions',
                   'Ask when ambiguous': 'ask'}

st.set_page_config(page_title='Health Insurance Document Agent', layout='wide')
st.title('Health Insurance Document Agent')
st.caption('Choose your sources. Ask a question. Check the evidence.')
st.session_state.setdefault('messages', [])
st.session_state.setdefault('session_id', uuid.uuid4().hex)
configure_tracing()
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
    ambiguity = AMBIGUITY_MODES[st.selectbox(
        'Ambiguous questions', list(AMBIGUITY_MODES),
        help='Off answers directly. State assumptions answers the most likely reading and names it. '
             'Ask when ambiguous offers the possible readings for you to choose when they would give '
             'different answers. The last two add one model call per question.')]
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
    if result.get('status') == 'needs_clarification':
        return
    interpretation = result.get('interpretation')
    if interpretation and interpretation.get('alternatives'):
        st.caption('Answered as: ' + interpretation['used'])
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


def ask(question, clarification=None):
    """Run the agent. A clarification re-runs the original question with the reading the user chose."""
    shown = f'{question}\n\n*Meaning: {clarification}*' if clarification else question
    history = [{'role':m['role'],'content':m['content']} for m in st.session_state.messages]
    st.session_state.messages.append({'role':'user','content':shown})
    with st.chat_message('user'):
        st.markdown(shown)
    with st.chat_message('assistant'):
        status = st.empty()
        try:
            # A per-call model setting is passed to the agent; never share user file caches.
            result = answer_question(st.session_state.documents,selected,question,
                app='chat',
                session_id=st.session_state.session_id, history=history,
                on_step=lambda _:status.caption('Checking the selected sources…'),model=model,
                ambiguity=ambiguity, clarification=clarification)
            status.empty()
            st.markdown(result['answer'])
            show_evidence(result)
            st.caption(f"{result['usage']['llm_calls']} model calls · "
                       f"{result['usage']['input_tokens']:,} input tokens · "
                       f"reference {result['request_id'][:12]}")
            st.session_state.messages.append({'role':'assistant','content':result['answer'],'result':result,
                                              'question':question})
            if result.get('status') == 'needs_clarification':
                st.rerun()  # redraw so the reading buttons appear
        except Exception as exc:
            status.empty()
            if 'expired' in str(exc).lower() or 'LoginRefreshRequired' in type(exc).__name__:
                st.error('AWS login has expired. Renew your configured AWS profile, then try again.')
            else:
                st.error(f'The question could not be completed: {type(exc).__name__}: {exc}')


def pending_clarification():
    """The last assistant message, when it is an unanswered clarifying question."""
    messages = st.session_state.messages
    if messages and messages[-1].get('result', {}).get('status') == 'needs_clarification':
        return messages[-1]
    return None


for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message['role']):
        if message.get('result', {}).get('status') == 'needs_clarification':
            clarification = message['result']['clarification']
            st.markdown(clarification['question'])
            if clarification.get('why'):
                st.caption(clarification['why'])
            if message is pending_clarification():
                for number, option in enumerate(clarification['options']):
                    if st.button(option, key=f'reading-{index}-{number}'):
                        st.session_state.chosen_reading = (message['question'], option)
                        st.rerun()
                st.caption('Or type what you meant below.')
            else:
                st.markdown('\n'.join(f'- {option}' for option in clarification['options']))
            continue
        st.markdown(message['content'])
        if 'result' in message:
            show_evidence(message['result'])

waiting = pending_clarification()
typed = st.chat_input('Describe what you meant' if waiting else 'Ask about the selected files',
                      disabled=not selected)
chosen = st.session_state.pop('chosen_reading', None)
if chosen:
    ask(*chosen)
elif typed and waiting:
    ask(waiting['question'], clarification=typed)
elif typed:
    ask(typed)
