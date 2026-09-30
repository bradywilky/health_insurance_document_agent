"""Deterministic calculations, source coverage and answer checks; model replies are scripted."""
import json

import pytest

from backend.agents.answer_checks import answer_issues, payment_certainty, unsupported_amounts
from backend.agents.document_agent import run_document_agent
from backend.preprocessing.documents import ingest_document
from backend.tools.calculations import calculate, date_calculate

SOURCES = {'E1': 'Two completed sessions. Billed USD 220.00.', 'E2': 'Rate USD 88.00 per unit.',
           'E3': 'Coinsurance is 20 percent.', 'question': 'Service on 2026-08-15'}


def test_calculate_checks_literals_against_cited_sources():
    result = calculate({'label': 'allowed', 'expression': 'two * 88.00'.replace('two', '2'),
                        'sources': ['E1', 'E2']}, SOURCES)
    assert result['result'] == '176.00'
    coinsurance = calculate({'label': 'coinsurance', 'expression': '(176.00 - 50) * 20 / 100',
                             'sources': ['E4', 'E3']}, {**SOURCES, 'E4': json.dumps(result) + ' deductible 50'})
    assert coinsurance['result'] == '25.20'


@pytest.mark.parametrize('expression,sources,match', [
    ('2 * 80', ['E1', 'E2'], 'do not appear'),
    ('2 * 88', ['E9'], 'Unknown sources'),
    ('__import__("os")', ['E2'], 'support'),
    ('88 ** 2', ['E2'], 'support'),
    ('88 / 0', ['E2'], 'zero'),
    ('88 *', ['E2'], 'arithmetic'),
])
def test_calculate_rejects_unverified_or_unsafe_requests(expression, sources, match):
    with pytest.raises(ValueError, match=match):
        calculate({'label': 'x', 'expression': expression, 'sources': sources}, SOURCES)


def test_date_calculations():
    assert date_calculate({'label': 'deadline', 'operation': 'add_days', 'date': '2026-04-05',
                           'days': 15})['result'] == '2026-04-20'
    assert date_calculate({'label': 'effective', 'operation': 'compare', 'date': '2026-08-15',
                           'other_date': '2026-07-01'})['result'] == 'after'
    with pytest.raises(ValueError, match='ISO'):
        date_calculate({'label': 'x', 'operation': 'compare', 'date': 'August 15', 'other_date': '2026-07-01'})


def test_answer_checks_flag_unsupported_amounts_and_payment_certainty():
    evidence = [{'id': 'E1', 'data': {'text': 'Rate USD 88.00; two units.'}},
                {'id': 'E2', 'tool': 'calculate', 'data': {'result': '176.00'}}]
    assert unsupported_amounts('Scheduled USD 176 and $85.20 member.', evidence, 'Rate?') == ['85.20']
    assert payment_certainty('The insurer will pay USD 176.') == ['The insurer will pay USD 176.']
    assert payment_certainty('We cannot determine what the insurer will pay.') == []
    assert answer_issues('Scheduled amount USD 176.00 [E2].', evidence, 'Rate?') == []


def two_docs():
    rates = ingest_document('rates.txt', b'Rate USD 88.00 per unit.')
    other = ingest_document('benefits.txt', b'Coinsurance is 20 percent.')
    return rates, other, {rates.id: rates, other.id: other}


def test_coverage_prompt_then_examined(native_script):
    rates, other, docs = two_docs()
    requests = native_script([
        {'tool': 'read_document', 'parameters': {'document_id': rates.id}},
        {'tool': 'answer', 'parameters': {}},
        {'tool': 'read_document', 'parameters': {'document_id': other.id}},
        {'tool': 'answer', 'parameters': {}}, 'Rate USD 88.00 [E1].'])
    result = run_document_agent(docs, list(docs), 'Rate?')
    prompt = requests[2]['messages'][-1]['content'][0]['toolResult']['content'][0]['text']
    assert 'benefits.txt' in prompt and 'not examined' in prompt
    assert result['protocol']['coverage_prompts'] == 1
    assert result['coverage'] == {'examined': ['rates.txt', 'benefits.txt'], 'auto_read': [], 'unexamined': []}
    assert not any('Not examined' in w for w in result['limitations'])


def test_ignored_coverage_prompt_auto_reads_short_documents(native_script):
    rates, other, docs = two_docs()
    requests = native_script([
        {'tool': 'read_document', 'parameters': {'document_id': rates.id}},
        {'tool': 'answer', 'parameters': {}}, {'tool': 'answer', 'parameters': {}},
        'Rate USD 88.00 [E1]; coinsurance 20 percent [E2].'])
    result = run_document_agent(docs, list(docs), 'Rate?')
    assert result['coverage']['auto_read'] == ['benefits.txt'] and result['coverage']['unexamined'] == []
    assert result['evidence'][1]['auto'] is True and result['protocol']['auto_reads'] == 1
    writer = json.loads(requests[-1]['messages'][0]['content'][0]['text'])
    assert 'Coinsurance is 20 percent' in json.dumps(writer['evidence'])


def test_long_unread_document_is_disclosed_not_auto_read(native_script):
    rates, _, _ = two_docs()
    long_doc = ingest_document('manual.txt', '\n\n'.join(f'Section {i} text.' for i in range(40)).encode())
    docs = {rates.id: rates, long_doc.id: long_doc}
    assert len(long_doc.blocks) > 20
    requests = native_script([
        {'tool': 'read_document', 'parameters': {'document_id': rates.id}},
        {'tool': 'answer', 'parameters': {}}, {'tool': 'answer', 'parameters': {}}, 'Rate USD 88.00 [E1].'])
    result = run_document_agent(docs, list(docs), 'Rate?')
    assert result['coverage']['unexamined'] == ['manual.txt']
    assert any('Not examined during research: manual.txt' in w for w in result['limitations'])
    assert json.loads(requests[-1]['messages'][0]['content'][0]['text'])['unexamined_documents'] == ['manual.txt']


def test_calculation_evidence_reaches_writer_and_bad_inputs_are_not_protocol_failures(native_script):
    rates, _, _ = two_docs()
    requests = native_script([
        {'tool': 'read_document', 'parameters': {'document_id': rates.id}},
        {'tool': 'calculate', 'parameters': {'label': 'scheduled', 'expression': '2 * 80', 'sources': ['E1', 'question']}},
        {'tool': 'calculate', 'parameters': {'label': 'scheduled', 'expression': '2 * 88.00', 'sources': ['E1', 'question']}},
        {'tool': 'answer', 'parameters': {}}, 'Scheduled amount USD 176.00 [E2].'])
    result = run_document_agent({rates.id: rates}, [rates.id], 'Scheduled amount for 2 units?')
    rejected = requests[2]['messages'][-1]['content'][0]['toolResult']['content'][0]['text']
    assert 'do not appear' in rejected
    assert result['protocol']['repair_attempts'] == 0
    assert result['evidence'][1]['data']['result'] == '176.00'
    writer = json.loads(requests[-1]['messages'][0]['content'][0]['text'])
    assert any(e['tool'] == 'calculate' for e in writer['evidence'])
    assert not result['limitations']


def test_uncomputed_amount_returns_to_planner_once(native_script):
    rates, _, _ = two_docs()
    requests = native_script([
        {'tool': 'read_document', 'parameters': {'document_id': rates.id}},
        {'tool': 'answer', 'parameters': {}},
        'Scheduled amount USD 176.00 [E1].',
        {'tool': 'calculate', 'parameters': {'label': 'scheduled', 'expression': '2 * 88.00', 'sources': ['E1', 'question']}},
        {'tool': 'answer', 'parameters': {}},
        'Scheduled amount USD 176.00 [E2].'])
    result = run_document_agent({rates.id: rates}, [rates.id], 'Scheduled amount for 2 units?')
    last = requests[3]['messages'][-1]['content']
    assert all('toolResult' in block for block in last)
    note = last[-1]['toolResult']['content'][-1]['text']
    assert '176.00' in note and 'calculate' in note
    assert [m['role'] for m in requests[3]['messages']] == ['user', 'assistant', 'user', 'assistant', 'user']
    assert result['protocol']['calculation_prompts'] == 1
    assert result['answer'] == 'Scheduled amount USD 176.00 [E2].'
    assert not result['limitations']


def test_flagged_answer_gets_one_revision(native_script):
    rates, _, _ = two_docs()
    requests = native_script([
        {'tool': 'read_document', 'parameters': {'document_id': rates.id}},
        {'tool': 'answer', 'parameters': {}},
        'The insurer will pay USD 88.00 per unit [E1].', 'Scheduled amount is USD 88.00 per unit [E1].'])
    result = run_document_agent({rates.id: rates}, [rates.id], 'Rate?')
    revision = json.loads(requests[-1]['messages'][0]['content'][0]['text'])
    assert len(revision['issues']) == 1
    assert result['answer'].startswith('Scheduled amount is USD 88.00')
    assert result['protocol']['answer_revisions'] == 1
    assert not any('Answer check' in w for w in result['limitations'])


def test_unfixed_answer_issues_become_limitations(native_script):
    rates, _, _ = two_docs()
    native_script([{'tool': 'read_document', 'parameters': {'document_id': rates.id}},
                   {'tool': 'answer', 'parameters': {}},
                   'The insurer will pay USD 88.00 [E1].', 'The insurer will pay USD 88.00 [E1].'])
    result = run_document_agent({rates.id: rates}, [rates.id], 'Rate?')
    assert any('Answer check: States a payment as certain' in w for w in result['limitations'])


def test_numbers_inside_dates_do_not_verify_calculations():
    sources = {'E1': 'Receipt date August 20, 2026; service 2026-08-15. Deductible USD 50.00.'}
    with pytest.raises(ValueError, match="'20'"):
        calculate({'label': 'coinsurance', 'expression': '(176 - 50) * 20 / 100', 'sources': ['E1']},
                  {**sources, 'E1': sources['E1'] + ' allowed 176'})
    assert calculate({'label': 'x', 'expression': '176 - 50.00', 'sources': ['E1']},
                     {'E1': sources['E1'] + ' allowed 176'})['result'] == '126.00'
