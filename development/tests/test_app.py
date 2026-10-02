from pathlib import Path
import pytest

from streamlit.testing.v1 import AppTest
from backend.preprocessing.documents import ingest_document
from backend.storage.s3 import configured_store

def saved(name, content):
    return ingest_document(name, content, store=configured_store())

def key(doc):
    return doc.storage_ref["manifest_key"]

APP = Path(__file__).resolve().parents[2] / 'apps' / 'chat' / 'app.py'


@pytest.fixture(autouse=True)
def local_storage_for_tests(monkeypatch, tmp_path):
    monkeypatch.setenv('DOCUMENTS_STORAGE', 'local')
    monkeypatch.setenv('DOCUMENTS_LOCAL_DIR', str(tmp_path))


def test_empty_app_requires_selection():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert app.chat_input[0].disabled
    assert app.multiselect[0].value == []


def test_chat_receives_selected_files_and_selection_change_clears_history(monkeypatch):
    first = saved('first.txt',b'Rate is USD 100.')
    second = saved('second.txt',b'Rate is USD 200.')
    calls=[]
    def fake_run(docs,ids,question,**kwargs):
        calls.append((ids,question))
        return {'answer':'Selected rate is USD 100 [E1].','evidence':[], 'limitations':[]}
    monkeypatch.setattr('backend.agents.document_agent.run_document_agent',fake_run)
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.session_state['documents'] = {first.id:first,second.id:second}
    app.run()
    app.multiselect[0].set_value([key(first)]).run()
    assert not app.chat_input[0].disabled
    app.chat_input[0].set_value('What is the rate?').run(timeout=20)
    assert not app.exception
    assert calls == [([first.id],'What is the rate?')]
    assert len(app.session_state['messages']) == 2
    app.multiselect[0].set_value([key(second)]).run()
    assert app.session_state['messages'] == []


def test_chat_shows_each_model_call_from_the_local_capture(native_script):
    doc = saved('rates.txt', b'Plan Alpha member rate is USD 88 per month.')
    native_script([{'tool': 'search_documents', 'parameters': {'query': 'Alpha rate', 'document_ids': [doc.id]}},
                   {'tool': 'answer', 'parameters': {}}, 'The member rate is USD 88 [E1].'])
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.multiselect[0].set_value([key(doc)]).run()
    app.chat_input[0].set_value('What is the member rate for plan Alpha?').run(timeout=20)
    assert not app.exception
    panel = next(e for e in app.expander if e.label.startswith('Model calls'))
    assert panel.label == 'Model calls (3 · 45 tokens)'
    assert [m.value for m in panel.metric][:3] == ['3', '30', '15']
    picker = panel.selectbox[0]
    assert picker.options == ['#1 Native Document Research (15 tokens)', '#2 Native Document Research (15 tokens)',
                              '#3 Document Answer (15 tokens)']
    picker.set_value(3).run()
    panel = next(e for e in app.expander if e.label.startswith('Model calls'))
    assert any('member rate is USD 88' in c.value for c in panel.code)  # the chosen call's response


def test_clear_removes_session_documents():
    doc = saved('first.txt',b'Rate is USD 100.')
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.session_state['documents'] = {doc.id:doc}
    app.run()
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state['documents'] == {}
    assert len(configured_store().list_documents()) == 1


def test_upload_portal_and_independent_chat_share_saved_examples():
    portal = AppTest.from_file(str(APP.parents[1] / 'upload' / 'app.py')).run(timeout=20)
    assert not portal.exception
    assert not portal.chat_input
    assert len(portal.get('file_uploader')) == 1
    next(b for b in portal.button if b.label == 'Load Bingle-Dingle examples').click().run(timeout=20)
    assert not portal.exception
    assert len(portal.success) == 8
    chat = AppTest.from_file(str(APP)).run(timeout=20)
    assert not chat.exception
    assert not chat.get('file_uploader')
    assert len(chat.multiselect[0].options) == 8
    items = configured_store().list_documents()
    chat.multiselect[0].set_value([items[0]['manifest_key']]).run()
    assert not chat.chat_input[0].disabled


def test_replacement_clears_old_conversation():
    doc = saved('policy.txt', b'Old rate 100')
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.multiselect[0].set_value([key(doc)]).run()
    app.session_state['messages'] = [{'role':'assistant', 'content':'Old answer'}]
    replacement = saved('policy.txt', b'New rate 200')
    app.run()
    assert app.session_state['messages'] == []
    assert list(app.session_state['documents']) == [replacement.id]


def test_model_switch_clears_history(monkeypatch):
    calls=[]
    def fake_run(docs,ids,question,**kwargs):
        calls.append(kwargs)
        return {'answer':'Rate USD 100 [E1].','evidence':[],'limitations':[]}
    monkeypatch.setattr('backend.agents.document_agent.run_document_agent',fake_run)
    app=AppTest.from_file(str(APP)).run(timeout=20)
    doc=saved('rates.txt',b'Rate USD 100.')
    app.session_state['documents']={doc.id:doc}
    app.run()
    app.multiselect[0].set_value([key(doc)]).run()
    next(s for s in app.selectbox if s.label=='Model').set_value('claude-sonnet-4.5').run()
    app.chat_input[0].set_value('Rate?').run(timeout=20)
    assert calls[-1]['model']=='claude-sonnet-4.5'
    next(s for s in app.selectbox if s.label=='Model').set_value('claude-haiku-4.5').run()
    assert app.session_state['messages']==[]
    assert not app.exception
