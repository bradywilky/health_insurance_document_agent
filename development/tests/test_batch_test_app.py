import io
from pathlib import Path
import pytest

from streamlit.testing.v1 import AppTest
from apps.batch_test.app import read_questions, run_batch
from backend.preprocessing.documents import ingest_document
from backend.storage.s3 import configured_store

APP = Path(__file__).resolve().parents[2] / 'apps' / 'batch_test' / 'app.py'


@pytest.fixture(autouse=True)
def local_storage_for_tests(monkeypatch, tmp_path):
    monkeypatch.setenv('DOCUMENTS_STORAGE', 'local')
    monkeypatch.setenv('DOCUMENTS_LOCAL_DIR', str(tmp_path))


def test_read_questions_skips_header_and_blanks():
    csv = io.StringIO('question\nWhat is the rate?\n\n"Who pays, and when?"\n')
    assert read_questions(csv) == ['What is the rate?', 'Who pays, and when?']


def test_run_batch_has_answer_and_metric_columns_per_model(native_script):
    doc = ingest_document('rates.txt', b'Plan Alpha member rate is USD 88 per month.', store=configured_store())
    turns = [{'tool': 'search', 'parameters': {'query': 'Alpha rate'}}, {'tool': 'answer', 'parameters': {}},
             'The member rate is USD 88 [E1].']
    native_script(turns * 2)
    results = run_batch(['What is the member rate for plan Alpha?'], [doc], ['maverick', 'nova-2-lite'])
    row = results.iloc[0]
    for model in ('maverick', 'nova-2-lite'):
        assert row[f'{model} status'] == 'answered'
        assert 'USD 88' in row[f'{model} answer']
        assert row[f'{model} llm_calls'] == 3
        assert row[f'{model} input_tokens'] == 30 and row[f'{model} output_tokens'] == 15
        assert row[f'{model} latency_s'] >= 0


def test_app_run_button_needs_questions_documents_and_models():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert app.button[0].disabled
