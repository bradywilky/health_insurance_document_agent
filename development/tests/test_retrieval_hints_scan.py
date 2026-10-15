"""Search hints (upload-time other words per block) and the relevance scan tool. Model replies are scripted."""
import json
import threading

import pytest

from backend.agents.document_agent import run_document_agent
from backend.agents.document_protocol import examined_documents
from backend.agents.llm import Usage
from backend.agents.search_hints import generate_hints
from backend.preprocessing.documents import ingest_document
from backend.retrieval.scan import scan_documents
from backend.retrieval.search import search_documents
from backend.storage.filesystem import LocalDocumentStore

BENEFITS = (b'Emergency room services are covered after a USD 150 copay.\n\n'
            b'Ambulance transport requires prior authorization.\n\n'
            b'Coverage ends on the last day of the month after termination.')


class ByPayload:
    """Stands in for LLM.converse from several threads: replies with reply(payload) as the one tool call."""
    model_id = 'test-model'

    def __init__(self, reply):
        self.reply, self.payloads, self.lock = reply, [], threading.Lock()

    def converse(self, system, messages, tool_config=None):
        payload = json.loads(messages[0]['content'][0]['text'])
        with self.lock:
            self.payloads.append(payload)
        result = self.reply(payload)
        if isinstance(result, Exception):
            raise result
        name = tool_config['tools'][0]['toolSpec']['name']
        return {'stopReason': 'tool_use', 'output': {'message': {'role': 'assistant', 'content': [
            {'toolUse': {'toolUseId': 't1', 'name': name, 'input': result}}]}}}


def hints_for(payload):
    known = {'Emergency': ['ER visit', 'urgent care'], 'Ambulance': ['PA', 'preauthorization']}
    return {'passages': [{'block_id': p['block_id'], 'hints': hints} for p in payload['passages']
                         for word, hints in known.items() if word in p['text']]}


def test_search_finds_blocks_through_hints_but_returns_their_own_text(tmp_path):
    plain = ingest_document('benefits.txt', BENEFITS)
    assert search_documents({plain.id: plain}, 'ER visit')['matches'] == []
    store = LocalDocumentStore(tmp_path)
    doc = ingest_document('benefits.txt', BENEFITS, store=store, hint=lambda d: generate_hints(d, ByPayload(hints_for)))
    assert doc.search_hints['meta']['blocks_with_hints'] == 2
    loaded = store.load(doc.storage_ref['manifest_key'])
    assert loaded.search_hints == doc.search_hints
    result = search_documents({loaded.id: loaded}, 'ER visit')
    [match] = result['matches']
    assert match['block_id'] == 'B1' and match['matched_hints'] == ['er', 'visit']
    assert match['text'].startswith('Emergency room') and 'hints' not in match
    assert 'search hints' in result['retrieval_method']
    # A word in the text and the hints counts once; text matches carry no matched_hints.
    assert 'matched_hints' not in search_documents({loaded.id: loaded}, 'ambulance')['matches'][0]


def test_hints_run_in_parallel_batches_and_failures_never_block_upload(tmp_path):
    text = '\n\n'.join(f'Paragraph {i} about Emergency care.' for i in range(40)).encode()
    doc = ingest_document('long.txt', text)
    def flaky(payload):
        return ValueError('throttled') if payload['passages'][0]['block_id'] == 'B16' else hints_for(payload)
    llm = ByPayload(flaky)
    hints = generate_hints(doc, llm, workers=3)
    assert len(llm.payloads) == 3 + 1  # three batches of 15; the failed one is retried once
    assert len(hints['blocks']) == 25 and hints['meta']['failed_batches'][0]['blocks'] == 'B16-B30'

    def broken(d):
        raise ValueError('model unavailable')
    store = LocalDocumentStore(tmp_path)
    saved = ingest_document('benefits.txt', BENEFITS, store=store, hint=broken)
    assert any('Search hints not generated' in w for w in saved.warnings)
    key = saved.storage_ref['manifest_key']
    assert store.load(key).search_hints is None
    store.save_search_hints(key, generate_hints(saved, ByPayload(hints_for)))
    assert store.load(key).search_hints['meta']['blocks_with_hints'] == 2
    with pytest.raises(PermissionError):
        LocalDocumentStore(tmp_path, read_only=True).save_search_hints(key, hints)


def relevant_when(word):
    def reply(payload):
        return {'relevant': [{'block_id': p['block_id'], 'reason': f'mentions {word}'}
                             for p in payload['passages'] if word in p['text']]}
    return reply


def test_scan_reads_every_block_in_parallel_and_keeps_document_order():
    paragraphs = [f'Paragraph {i}: routine text.' for i in range(45)]
    paragraphs[3] = 'Ambulance transport needs approval.'
    paragraphs[30] = 'Ambulance claims go to the medical unit.'
    paragraphs[44] = 'Ambulance mileage is paid per mile.'
    doc = ingest_document('guide.txt', '\n\n'.join(paragraphs).encode())
    other = ingest_document('benefits.txt', BENEFITS)
    llm = ByPayload(relevant_when('Ambulance'))
    result = scan_documents({doc.id: doc, other.id: other}, 'Is an ambulance covered?', llm, limit=2, workers=4)
    assert len(llm.payloads) == 4  # three batches of 20 blocks, plus one for the short document
    assert all(p['question'] == 'Is an ambulance covered?' for p in llm.payloads)
    assert [m['block_id'] for m in result['matches']] == ['B4', 'B31']
    assert result['matches'][0]['text'] == 'Ambulance transport needs approval.'
    assert result['matches'][0]['reason'] == 'mentions Ambulance'
    assert [(m['document_id'], m['block_id']) for m in result['more_relevant']] == [(doc.id, 'B45'), (other.id, 'B2')]
    assert result['total_relevant_blocks'] == 4 and result['truncated'] and result['blocks_scanned'] == 48
    assert set(result['scanned_document_ids']) == {doc.id, other.id}


def test_scan_reports_failed_batches_and_limits_its_size(monkeypatch):
    doc = ingest_document('benefits.txt', BENEFITS)
    result = scan_documents({doc.id: doc}, 'ambulance', ByPayload(lambda p: ValueError('throttled')))
    assert result['matches'] == [] and result['scanned_document_ids'] == []
    assert result['failed_batches'][0]['blocks'] == 'B1-B3'
    record = {'tool': 'scan', 'parameters': {}, 'data': result}
    assert examined_documents([record]) == set()
    # A completed scan examines a document even when nothing in it was relevant.
    clean = scan_documents({doc.id: doc}, 'dental', ByPayload(relevant_when('Dental')))
    assert examined_documents([{'tool': 'scan', 'parameters': {}, 'data': clean}]) == {doc.id}
    monkeypatch.setenv('SCAN_MAX_BLOCKS', '2')
    with pytest.raises(ValueError, match='at most 2 text blocks'):
        scan_documents({doc.id: doc}, 'ambulance', ByPayload(relevant_when('Ambulance')))


def test_agent_scan_tool_uses_the_model_and_counts_its_usage(native_script):
    doc = ingest_document('benefits.txt', BENEFITS)
    requests = native_script([
        {'tool': 'scan', 'parameters': {'query': 'ambulance approval rules'}},
        {'tool': 'record_relevant', 'parameters': {'relevant': [{'block_id': 'B2', 'reason': 'authorization rule'}]}},
        {'tool': 'answer', 'parameters': {}},
        'Ambulance transport requires prior authorization.'])
    usage = Usage()
    result = run_document_agent({doc.id: doc}, [doc.id], 'Do I need approval for an ambulance?', usage=usage)
    scan_request = requests[1]
    assert scan_request['toolConfig']['tools'][0]['toolSpec']['name'] == 'record_relevant'
    assert 'ambulance approval rules' in scan_request['messages'][0]['content'][0]['text']
    [evidence] = [e for e in result['evidence'] if e['tool'] == 'scan']
    assert [m['block_id'] for m in evidence['data']['matches']] == ['B2']
    assert usage.llm_calls == 4
