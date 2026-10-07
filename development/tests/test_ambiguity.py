"""Ambiguity modes (off / assumptions / ask), the ask policy, and derive grouping. Model replies are scripted."""
import json
from pathlib import Path

import pytest

from backend.agents.ambiguity import decide
from backend.agents.document_agent import run_document_agent
from backend.preprocessing.documents import ingest_document
from backend.shared.table import parse_csv
from backend.tools.table_operations import query_table

CSV = b'Element,Primary,Secondary\nAddress - Code,Out of Scope,Out of Scope\nAddress - Date,Out of Scope,Kept\n' \
      b'Customer - Code,Out of Scope,Out of Scope\nCustomer - Date,Kept,Kept\n'


QUESTION = 'What has the most out of scope?'


def assessment(primary, *alternatives, phrase='the most out of scope', question='Which do you mean?'):
    return {'ambiguous_phrase': phrase, 'primary_reading': primary, 'alternative_readings': list(alternatives),
            'why': 'Out of scope can be grouped several ways.', 'clarifying_question': question}


TWO = assessment('By element prefix', 'By application column')


def test_policy_asks_only_for_grounded_alternatives():
    asked = decide(TWO, 'ask', QUESTION)
    assert asked['decision'] == 'ask'
    assert asked['clarification']['options'] == ['By element prefix', 'By application column']
    assert asked['clarification']['why']
    assumed = decide(TWO, 'assumptions', QUESTION)
    assert assumed == {'decision': 'proceed', 'interpretation': {
        'used': 'By element prefix', 'alternatives': ['By application column']}}
    for clear in [assessment('Only reading'),
                  assessment('A', 'B', phrase=''),                   # no phrase named
                  assessment('A', 'B', phrase='invented wording'),   # phrase not in the question
                  assessment('A', 'a ')]:                            # a rewording is not an alternative
        result = decide(clear, 'ask', QUESTION)
        assert result['decision'] == 'proceed' and result['interpretation']['alternatives'] == []
    # Case and spacing differences in the quoted phrase are tolerated.
    assert decide(assessment('A', 'B', phrase='THE  most out of Scope'), 'ask', QUESTION)['decision'] == 'ask'


def test_text_documents_get_a_short_preview_tables_do_not():
    from backend.agents.ambiguity import PREVIEW_CHARS, document_view
    text = ingest_document('policy.txt', ('Allowed amount rules. ' * 200).encode())
    assert 0 < len(document_view(text)['preview']) <= PREVIEW_CHARS
    assert 'preview' not in document_view(ingest_document('mapping.csv', CSV))


@pytest.fixture
def doc():
    return ingest_document('mapping.csv', CSV)


def read(doc):
    return {'tool': 'table_tool', 'parameters': {'document_id': doc.id, 'name': 'read_sheet',
                                                 'parameters': {'sheet_name': 'Sheet1'}}}


def assess_call(value):
    return {'tool': 'assess_question', 'parameters': value}


def test_off_mode_makes_no_assessment_call(doc, native_script):
    requests = native_script([read(doc), {'tool': 'answer', 'parameters': {}}, 'Address has 2 [E1].'])
    result = run_document_agent({doc.id: doc}, [doc.id], 'What has the most out of scope?')
    assert result['status'] == 'answered' and result['interpretation'] is None
    assert result['protocol']['ambiguity_decision'] == 'off'
    assert all('assess_question' not in json.dumps(r.get('toolConfig', {})) for r in requests)


def test_ask_mode_returns_clarification_without_research(doc, native_script):
    requests = native_script([assess_call(TWO)])
    result = run_document_agent({doc.id: doc}, [doc.id], QUESTION, ambiguity='ask')
    assert result['status'] == 'needs_clarification' and result['evidence'] == []
    assert result['clarification']['options'] == ['By element prefix', 'By application column']
    assert '1. By element prefix' in result['answer']
    assert len(requests) == 1
    payload = json.loads(requests[0]['messages'][0]['content'][0]['text'])
    assert payload['selected_documents'][0]['filename'] == 'mapping.csv'


def test_assumptions_mode_passes_reading_to_planner_and_writer(doc, native_script):
    requests = native_script([assess_call(TWO), read(doc), {'tool': 'answer', 'parameters': {}},
                              'Reading: by element prefix. Address has 2 [E1].'])
    result = run_document_agent({doc.id: doc}, [doc.id], 'What has the most out of scope?',
                                ambiguity='assumptions')
    assert result['status'] == 'answered'
    assert result['interpretation'] == {'used': 'By element prefix', 'alternatives': ['By application column']}
    assert 'By element prefix' in requests[1]['messages'][0]['content'][-1]['text']
    writer = json.loads(requests[-1]['messages'][0]['content'][0]['text'])
    assert writer['interpretation']['alternatives'] == ['By application column']


def test_user_clarification_skips_assessment(doc, native_script):
    requests = native_script([read(doc), {'tool': 'answer', 'parameters': {}}, 'Address has 2 [E1].'])
    result = run_document_agent({doc.id: doc}, [doc.id], 'What has the most out of scope?', ambiguity='ask',
                                clarification='By element prefix')
    assert result['protocol']['ambiguity_decision'] == 'clarified_by_user'
    assert result['interpretation'] == {'used': 'By element prefix', 'alternatives': []}
    assert 'assess_question' not in json.dumps(requests[0].get('toolConfig', {}))


def test_failed_assessment_fails_open(doc, native_script):
    native_script(['I think it is ambiguous.', read(doc), {'tool': 'answer', 'parameters': {}}, 'Address [E1].'])
    result = run_document_agent({doc.id: doc}, [doc.id], 'What has the most out of scope?', ambiguity='ask')
    assert result['status'] == 'answered'
    assert result['protocol']['ambiguity_decision'] == 'assessment_failed'


def test_invalid_mode_rejected(doc):
    with pytest.raises(ValueError, match='ambiguity'):
        run_document_agent({doc.id: doc}, [doc.id], 'Q?', ambiguity='sometimes')


def frame():
    return parse_csv(CSV)


def test_derive_groups_by_part_of_a_value():
    result = query_table(frame(), {'derive': [{'column': 'Element', 'as': 'prefix', 'split': ' - ', 'part': 0}],
                                   'filters': [{'column': 'Primary', 'op': 'eq', 'value': 'Out of Scope'}],
                                   'group_by': ['prefix'], 'aggregations': [{'op': 'count_rows', 'as': 'n'}]})
    assert {r['prefix']: r['n'] for r in result['rows']} == {'Address': 2, 'Customer': 1}
    last = query_table(frame(), {'derive': [{'column': 'Element', 'as': 'kind', 'split': ' - ', 'part': -1}],
                                 'select': ['kind']})
    assert [r['kind'] for r in last['rows']] == ['Code', 'Date', 'Code', 'Date']
    missing = query_table(frame(), {'derive': [{'column': 'Element', 'as': 'x', 'split': ' - ', 'part': 5}],
                                    'filters': [{'column': 'x', 'op': 'is_null'}],
                                    'aggregations': [{'op': 'count_rows', 'as': 'n'}]})
    assert missing['rows'] == [{'n': 4}]


@pytest.mark.parametrize('spec,match', [
    ({'column': 'Element', 'as': 'Primary', 'split': '-'}, 'new column'),
    ({'column': 'Element', 'as': 'p', 'split': ''}, 'separator'),
    ({'column': 'Element', 'as': 'p', 'split': '-', 'part': '0'}, 'integer'),
    ({'column': 'Missing', 'as': 'p', 'split': '-'}, 'Unknown columns'),
])
def test_derive_validation(spec, match):
    with pytest.raises(ValueError, match=match):
        query_table(frame(), {'derive': [spec], 'select': ['Element']})


def test_chat_app_clarification_round_trip(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    from backend.storage.s3 import configured_store
    monkeypatch.setenv('DOCUMENTS_STORAGE', 'local')
    monkeypatch.setenv('DOCUMENTS_LOCAL_DIR', str(tmp_path))
    key = ingest_document('mapping.csv', CSV, store=configured_store()).storage_ref['manifest_key']
    calls = []

    def fake_run(docs, ids, question, **kwargs):
        calls.append(kwargs)
        if kwargs.get('ambiguity') == 'ask' and not kwargs.get('clarification'):
            clarification = {'question': 'Which do you mean?', 'options': ['By prefix', 'By column']}
            return {'status': 'needs_clarification', 'answer': 'Which do you mean?\n\n1. By prefix\n2. By column',
                    'clarification': clarification, 'evidence': [], 'limitations': [], 'protocol': {}}
        return {'status': 'answered', 'answer': 'Address has 2 [E1].', 'evidence': [], 'limitations': [],
                'interpretation': {'used': kwargs.get('clarification'), 'alternatives': []}}
    monkeypatch.setattr('backend.agents.document_agent.run_document_agent', fake_run)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / 'apps' / 'chat' / 'app.py')).run(timeout=20)
    app.multiselect[0].set_value([key]).run()
    next(s for s in app.selectbox if s.label == 'Ambiguous questions').set_value('Ask when ambiguous').run()
    app.chat_input[0].set_value('What has the most out of scope?').run(timeout=20)
    assert calls[-1]['ambiguity'] == 'ask' and calls[-1]['clarification'] is None
    next(b for b in app.button if b.label == 'By prefix').click().run(timeout=20)
    assert not app.exception
    assert calls[-1]['clarification'] == 'By prefix'
    assert app.session_state['messages'][-1]['content'] == 'Address has 2 [E1].'


def test_writer_hears_about_readings_only_when_one_was_chosen(doc, native_script):
    requests = native_script([read(doc), {'tool': 'answer', 'parameters': {}}, 'Address has 2 [E1].'])
    run_document_agent({doc.id: doc}, [doc.id], QUESTION)
    assert 'Answer the reading in' not in requests[-1]['system'][0]['text']
    requests = native_script([assess_call(TWO), read(doc), {'tool': 'answer', 'parameters': {}}, 'By prefix [E1].'])
    run_document_agent({doc.id: doc}, [doc.id], QUESTION, ambiguity='assumptions')
    assert 'Answer the reading in' in requests[-1]['system'][0]['text']
