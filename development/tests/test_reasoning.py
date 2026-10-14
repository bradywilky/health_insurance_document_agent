import json

from backend.preprocessing.documents import ingest_document
from backend.agents import reasoning
from backend.agents.document_agent import run_document_agent


def test_reasoning_runs_before_each_planner_call(native_script):
    doc = ingest_document('rates.txt', b'Rate USD 88.')
    requests = native_script(['Nothing yet. Read rates.txt because it is the only file.',
                              {'tool': 'read_document', 'parameters': {'document_id': doc.id}},
                              'E1 shows the rate. Call answer.',
                              {'tool': 'answer', 'parameters': {}}, 'Rate USD 88.'])
    r = run_document_agent({doc.id: doc}, [doc.id], 'Rate?', reasoning=True)
    assert [s['step'] for s in r['reasoning']] == [1, 2]
    assert r['reasoning'][0]['plan'].startswith('Nothing yet')
    assert r['protocol']['reasoning_steps'] == 2 and r['answer'] == 'Rate USD 88.'
    # The reasoning call has no tools; the planner call after it carries the plan.
    assert 'toolConfig' not in requests[0] and 'REASONING STEP' in requests[0]['system'][0]['text']
    assert 'Read rates.txt' in json.dumps(requests[1]['messages'])
    # The second reasoning call sees the first tool call and its result as text.
    transcript = requests[2]['messages'][0]['content'][0]['text']
    assert 'PLANNER CALLED read_document' in transcript and 'RESULT:' in transcript
    # Earlier plans are left out, so the model does not repeat its first plan.
    assert 'Read rates.txt' not in transcript


def test_reasoning_off_leaves_research_unchanged(native_script, monkeypatch):
    monkeypatch.delenv('PLANNER_REASONING', raising=False)
    doc = ingest_document('rates.txt', b'Rate USD 88.')
    requests = native_script([{'tool': 'read_document', 'parameters': {'document_id': doc.id}},
                              {'tool': 'answer', 'parameters': {}}, 'Rate USD 88.'])
    r = run_document_agent({doc.id: doc}, [doc.id], 'Rate?')
    assert 'reasoning' not in r and len(requests) == 3
    assert not any('next step should be' in json.dumps(q['messages']) for q in requests)


def test_setting_follows_environment_unless_explicit(monkeypatch):
    monkeypatch.setenv('PLANNER_REASONING', 'true')
    assert reasoning.enabled() and not reasoning.enabled(False)
    monkeypatch.setenv('PLANNER_REASONING', 'no')
    assert not reasoning.enabled() and reasoning.enabled(True)
