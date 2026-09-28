from pathlib import Path

from streamlit.testing.v1 import AppTest
from documents import ingest_document

APP = Path(__file__).resolve().parents[1] / 'app.py'


def test_empty_app_requires_selection():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert app.chat_input[0].disabled
    assert app.multiselect[0].value == []


def test_chat_receives_selected_files_and_selection_change_clears_history(monkeypatch):
    first = ingest_document('first.txt',b'Rate is USD 100.')
    second = ingest_document('second.txt',b'Rate is USD 200.')
    calls=[]
    def fake_run(docs,ids,question,**kwargs):
        calls.append((ids,question))
        return {'answer':'Selected rate is USD 100 [E1].','evidence':[], 'limitations':[]}
    monkeypatch.setattr('health_insurance_document_agent.run_document_agent',fake_run)
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.session_state['documents'] = {first.id:first,second.id:second}
    app.run()
    app.multiselect[0].set_value([first.id]).run()
    assert not app.chat_input[0].disabled
    app.chat_input[0].set_value('What is the rate?').run(timeout=20)
    assert not app.exception
    assert calls == [([first.id],'What is the rate?')]
    assert len(app.session_state['messages']) == 2
    app.multiselect[0].set_value([second.id]).run()
    assert app.session_state['messages'] == []


def test_clear_removes_session_documents():
    doc = ingest_document('first.txt',b'Rate is USD 100.')
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.session_state['documents'] = {doc.id:doc}
    app.run()
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state['documents'] == {}


def test_provider_examples_load_selected_sources_without_model_calls():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    next(button for button in app.button if button.label == 'Load Bingle-Dingle examples').click().run()
    assert not app.exception
    assert len(app.session_state['documents']) == 8
    assert set(app.multiselect[0].value) == set(app.session_state['documents'])
    assert not app.chat_input[0].disabled
    assert app.session_state['messages'] == []
