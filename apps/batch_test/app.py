"""Batch questions across models. Run: streamlit run apps/batch_test/app.py --server.port 8503

Upload a one-column CSV of questions, choose saved documents and models, and download a CSV with one row per
question and, for each model, its answer, status, latency and token counts.
"""
import io
import time
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import streamlit as st
from backend.entrypoint import ask_question_local
from backend.storage.s3 import configured_store
from backend.config.settings import MODELS

METRICS = ('status', 'latency_s', 'llm_calls', 'input_tokens', 'output_tokens')


def read_questions(upload):
    """Questions from the first column of a CSV, skipping blanks and a 'question' or 'questions' header row."""
    frame = pd.read_csv(upload, header=None, dtype=str, keep_default_na=False)
    questions = [q.strip() for q in frame.iloc[:, 0] if q.strip()]
    if questions and questions[0].lower() in {'question', 'questions'}:
        questions = questions[1:]
    return questions


def answer_one(question, docs, model):
    """One question through the local entry point, with wall-clock latency and token usage."""
    start = time.perf_counter()
    response = ask_question_local(question, documents=docs, model=model, app_id='batch_test')
    latency = time.perf_counter() - start
    usage = response['usage']
    return {'answer': response['answer'] or response['message'], 'status': response['status'],
            'latency_s': round(latency, 2), 'llm_calls': usage['llm_calls'],
            'input_tokens': usage['input_tokens'], 'output_tokens': usage['output_tokens']}


def run_batch(questions, docs, models, on_progress=None):
    """One row per question; columns '<model> answer' and '<model> <metric>' for each model."""
    rows, total, done = [], len(questions) * len(models), 0
    for question in questions:
        row = {'question': question}
        for model in models:
            result = answer_one(question, docs, model)
            row[f'{model} answer'] = result['answer']
            row.update({f'{model} {metric}': result[metric] for metric in METRICS})
            done += 1
            if on_progress:
                on_progress(done, total, question, model)
        rows.append(row)
    return pd.DataFrame(rows)


def summary(results, models):
    """Per-model totals and averages for a quick comparison."""
    return pd.DataFrame([{
        'model': model,
        'answered': int((results[f'{model} status'] == 'answered').sum()),
        'mean latency_s': round(results[f'{model} latency_s'].mean(), 2),
        'total latency_s': round(results[f'{model} latency_s'].sum(), 2),
        'mean llm_calls': round(results[f'{model} llm_calls'].mean(), 2),
        'total input_tokens': int(results[f'{model} input_tokens'].sum()),
        'total output_tokens': int(results[f'{model} output_tokens'].sum()),
    } for model in models])


st.set_page_config(page_title='Batch Test', layout='wide')
st.title('Batch Test')
st.caption('Run a list of questions against several models and download the answers with latency and tokens.')
try:
    store = configured_store(read_only=True)
    saved = {item['manifest_key']: item for item in store.list_documents()}
except Exception as exc:
    st.error(f'Cannot list saved documents: {exc}')
    st.stop()

upload = st.file_uploader('Questions CSV', type=['csv'],
                          help='One column, one question per row. A "question" header row is ignored.')
keys = st.multiselect('Files to answer from', list(saved), format_func=lambda key: saved[key]['filename'],
                      max_selections=10, help='Add files through the upload portal.')
models = st.multiselect('Models', list(MODELS), default=list(MODELS)[:1])

questions = []
if upload:
    try:
        questions = read_questions(upload)
    except Exception as exc:
        st.error(f'Could not read the CSV: {exc}')
    if questions:
        st.caption(f'{len(questions)} questions loaded.')
        with st.expander('Preview questions'):
            st.dataframe(pd.DataFrame({'question': questions}), hide_index=True)

ready = bool(questions and keys and models)
if st.button('Run', type='primary', disabled=not ready):
    try:
        docs = [store.load(key) for key in keys]
    except Exception as exc:
        st.error(f'Could not open saved files: {exc}')
        st.stop()
    progress = st.progress(0.0, text='Starting…')
    st.session_state.batch_results = run_batch(
        questions, docs, models,
        on_progress=lambda done, total, question, model: progress.progress(
            done / total, text=f'{done}/{total} · {model} · {question[:80]}'))
    st.session_state.batch_models = models
    progress.empty()

if 'batch_results' in st.session_state:
    results, ran = st.session_state.batch_results, st.session_state.batch_models
    st.subheader('Summary')
    st.dataframe(summary(results, ran), hide_index=True)
    st.subheader('Results')
    st.dataframe(results, hide_index=True)
    buffer = io.StringIO()
    results.to_csv(buffer, index=False)
    st.download_button('Download CSV', buffer.getvalue(), file_name='batch_test_results.csv', mime='text/csv')
st.caption('Questions and selected excerpts are sent to Amazon Bedrock. Each question is answered independently.')
