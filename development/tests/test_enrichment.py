"""Semantic profiles: grounding rules, chunk merging, repair, storage, planner hints. Model replies are scripted."""
import json

import pytest

from backend.agents.enrichment import CHUNK_CHARS, enrich, ground, make_enricher
from backend.preprocessing.documents import ingest_document
from backend.shared.profile import profile_hints
from backend.storage.filesystem import LocalDocumentStore

AMENDMENT = (b'AMENDMENT BD-AMD-2026-02\n\nStatus: EXECUTED. This amendment amends rate schedule BD-RATES-2026-01.\n\n'
             b'Effective July 1, 2026 the DEMO-SESSION rate is USD 88.00.\n\nPOS means place of service.')


def profile(**overrides):
    base = {'document_type': 'amendment', 'summary': 'Changes a rate.', 'status': 'executed',
            'status_quote': 'Status: EXECUTED.',
            'effective_dates': [{'label': 'effective', 'value': '2026-07-01', 'quote': 'Effective July 1, 2026'},
                                {'label': 'end', 'value': '2026-12-31', 'quote': 'ends December 31, 2026'}],
            'identifiers': [{'identifier': 'BD-AMD-2026-02', 'quote': 'AMENDMENT BD-AMD-2026-02'}],
            'references': [{'target': 'BD-RATES-2026-01', 'relationship': 'amends',
                            'quote': 'amends rate schedule BD-RATES-2026-01'},
                           {'target': 'BD-BEN-2026', 'relationship': 'depends_on', 'quote': 'see benefits BD-BEN-2026'}],
            'glossary': [{'term': 'POS', 'meaning': 'place of service', 'source': 'defined',
                          'quote': 'POS means place of service'},
                         {'term': 'DEMO-SESSION', 'meaning': 'a therapy session', 'source': 'defined',
                          'quote': 'DEMO-SESSION means therapy'},
                         {'term': 'PA', 'meaning': 'prior authorization', 'source': 'inferred', 'quote': ''}],
            'columns': [], 'status_values': []}
    base.update(overrides)
    return base


def test_grounding_keeps_quoted_claims_and_downgrades_or_drops_the_rest():
    doc = ingest_document('amendment.txt', AMENDMENT)
    result = ground(profile(), doc)
    assert result['status'] == 'executed' and result['status_location'] == {'paragraph': 2}
    assert [d['value'] for d in result['effective_dates']] == ['2026-07-01']
    assert [r['target'] for r in result['references']] == ['BD-RATES-2026-01']
    glossary = {g['term']: g['source'] for g in result['glossary']}
    # Quoted definition kept; unquoted "definition" downgraded; a term absent from the text dropped.
    assert glossary == {'POS': 'defined', 'DEMO-SESSION': 'inferred'}
    assert result['dropped'] == {'effective_dates': 1, 'identifiers': 0, 'references': 1, 'status_values': 0,
                                 'glossary_terms_not_in_document': 1, 'glossary_defined_downgraded': 1,
                                 'glossary_circular_guesses': 0, 'columns': 0}


def test_ungrounded_status_becomes_unknown():
    doc = ingest_document('amendment.txt', AMENDMENT)
    assert ground(profile(status='draft', status_quote='Status: DRAFT'), doc)['status'] == 'unknown'


def test_spreadsheet_columns_must_exist():
    doc = ingest_document('map.csv', b'Legacy Name,New Name\nA - Code,Out of Scope\nB - Code,B Code\n')
    result = ground(profile(glossary=[], columns=[
        {'sheet': 'Sheet1', 'column': 'New Name', 'role': 'name', 'note': 'Out of Scope when unmapped'},
        {'sheet': 'Sheet1', 'column': 'Invented', 'role': 'code'}]), doc)
    assert [c['column'] for c in result['columns']] == ['New Name']
    # Values listed in the column profile count as document text for quotes.
    assert ground(profile(status_values=[{'value': 'Out of Scope', 'meaning': 'not migrated',
                                          'quote': 'Out of Scope'}]), doc)['status_values']


class Scripted:
    """Stands in for LLM.converse: returns record_profile calls (or text) in order."""
    model_id = 'test-model'

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def converse(self, system, messages, tool_config=None):
        self.requests.append(json.loads(json.dumps(messages)))
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            return {'stopReason': 'end_turn', 'output': {'message': {'role': 'assistant', 'content': [{'text': reply}]}}}
        return {'stopReason': 'tool_use', 'output': {'message': {'role': 'assistant', 'content': [
            {'toolUse': {'toolUseId': f'p{len(self.requests)}', 'name': 'record_profile', 'input': reply}}]}}}


def test_enrich_retries_once_then_records_metadata():
    doc = ingest_document('amendment.txt', AMENDMENT)
    llm = Scripted('Here is the profile in prose.', profile())
    result = enrich(doc, llm)
    assert len(llm.requests) == 2 and result['meta']['model_id'] == 'test-model'
    assert result['meta']['coverage']['complete'] is True
    with pytest.raises(ValueError, match='No valid profile'):
        enrich(doc, Scripted('prose', {'document_type': 'x'}))


def test_long_documents_are_chunked_and_merged():
    paragraphs = [f'Paragraph {i}: ' + 'filler text ' * 120 for i in range(40)]
    paragraphs[1] = 'Status: EXECUTED.'
    paragraphs[30] = 'POS means place of service.'
    doc = ingest_document('long.txt', '\n\n'.join(paragraphs).encode())
    first = profile(glossary=[{'term': 'POS', 'meaning': 'point of sale', 'source': 'inferred', 'quote': ''}],
                    references=[], effective_dates=[], identifiers=[])
    later = profile(document_type='', summary='', glossary=[
        {'term': 'POS', 'meaning': 'place of service', 'source': 'defined', 'quote': 'POS means place of service'}],
        references=[], effective_dates=[], identifiers=[])
    count = -(-sum(len(b['text']) + 20 for b in doc.blocks) // CHUNK_CHARS)
    llm = Scripted(first, *[later] * 10)
    result = enrich(doc, llm)
    assert len(llm.requests) == count > 1
    # A later quoted definition replaces an earlier guess.
    assert result['glossary'] == [{'term': 'POS', 'meaning': 'place of service', 'source': 'defined',
                                   'quote': 'POS means place of service', 'location': {'paragraph': 31}}]


def test_enrichment_failure_never_blocks_upload(tmp_path):
    def broken(doc):
        raise ValueError('model unavailable')
    store = LocalDocumentStore(tmp_path)
    doc = ingest_document('amendment.txt', AMENDMENT, store=store, enrich=broken)
    assert doc.profile is None
    assert any('Semantic profile not generated' in w for w in doc.warnings)
    assert store.load(doc.storage_ref['manifest_key']).profile is None


def test_profile_storage_round_trip_and_review(tmp_path):
    store = LocalDocumentStore(tmp_path)
    doc = ingest_document('amendment.txt', AMENDMENT, store=store,
                          enrich=lambda d: enrich(d, Scripted(profile())))
    key = doc.storage_ref['manifest_key']
    loaded = store.load(key)
    assert loaded.profile['glossary'][0]['term'] == 'POS'
    reviewed = dict(loaded.profile, glossary=[{'term': 'PA', 'meaning': 'prior authorization',
                                               'source': 'confirmed', 'quote': '', 'location': None}])
    store.save_profile(key, reviewed)
    assert store.load(key).profile['glossary'][0]['source'] == 'confirmed'
    with pytest.raises(PermissionError):
        LocalDocumentStore(tmp_path, read_only=True).save_profile(key, reviewed)


def test_planner_summary_carries_compact_hints():
    doc = ingest_document('amendment.txt', AMENDMENT)
    doc.profile = ground(profile(), doc)
    doc.profile['glossary'].append({'term': 'DEMO', 'meaning': 'demo', 'source': 'confirmed'})
    hints = doc.summary()['semantic_profile']
    assert hints['references'] == ['amends BD-RATES-2026-01']
    # Model guesses stay out of the planner's view; confirmed terms come first.
    assert [g['source'] for g in hints['glossary']] == ['confirmed', 'defined']
    assert 'not evidence' in hints['note']
    assert 'semantic_profile' not in ingest_document('plain.txt', b'Plain text here.').summary()


def test_make_enricher_rejects_unknown_model():
    with pytest.raises(ValueError, match='Enrichment model'):
        make_enricher('gpt-unknown')


def test_upload_portal_toggle_and_review(monkeypatch, tmp_path):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('DOCUMENTS_STORAGE', 'local')
    monkeypatch.setenv('DOCUMENTS_LOCAL_DIR', str(tmp_path))
    monkeypatch.setattr('backend.agents.enrichment.make_enricher',
                        lambda *a, **k: (lambda doc: enrich(doc, Scripted(profile()))))
    app_path = Path(__file__).resolve().parents[2] / 'apps' / 'upload' / 'app.py'
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
    next(c for c in app.checkbox if 'semantic profile' in c.label).check().run(timeout=30)
    next(b for b in app.button if b.label == 'Load Bingle-Dingle examples').click().run(timeout=30)
    assert not app.exception
    store = LocalDocumentStore(tmp_path)
    saved = store.list_documents()
    assert all(item['has_profile'] for item in saved)
    key = next(i['manifest_key'] for i in saved if i['filename'] == 'bingle_dingle_amendment.txt')
    app.selectbox[0].set_value(key).run(timeout=30)
    next(b for b in app.button if b.label == 'Save reviewed profile').click().run(timeout=30)
    assert not app.exception
    assert store.load(key).profile['meta']['reviewed'] is True


def test_drafts_cannot_supersede_and_noise_is_removed():
    draft = (b'PROPOSAL BD-DRAFT-9 version 0.3\n\nStatus: DRAFT. Not executed; no contractual effect.\n\n'
             b'Would replace the rate in BD-AMD-2026-02.\n\nOut of Scope fields are listed below.')
    doc = ingest_document('draft.txt', draft)
    result = ground(profile(
        status='draft', status_quote='Status: DRAFT.',
        identifiers=[{'identifier': 'BD-DRAFT-9', 'quote': 'PROPOSAL BD-DRAFT-9'},
                     {'identifier': '0.3', 'quote': 'version 0.3'}],
        references=[{'target': 'BD-AMD-2026-02', 'relationship': 'supersedes',
                     'quote': 'Would replace the rate in BD-AMD-2026-02'}],
        glossary=[{'term': 'Out of Scope', 'meaning': 'fields that are out of scope', 'source': 'inferred', 'quote': ''}],
        effective_dates=[]), doc)
    assert [i['identifier'] for i in result['identifiers']] == ['BD-DRAFT-9']
    assert result['references'][0]['relationship'] == 'references'
    assert 'claimed supersedes' in result['references'][0]['note']
    assert result['glossary'] == [] and result['dropped']['glossary_circular_guesses'] == 1


def test_hints_show_document_wording_for_defined_terms():
    doc = ingest_document('amendment.txt', AMENDMENT)
    doc.profile = ground(profile(), doc)
    pos = next(g for g in doc.summary()['semantic_profile']['glossary'] if g['term'] == 'POS')
    assert pos['meaning'] == 'POS means place of service'


def test_spreadsheet_profiles_sample_data_rows_and_keep_narrative():
    from backend.agents.enrichment import SAMPLE_ROWS, _chunks
    rows = '\n'.join(f'R{i},Value {i}' for i in range(SAMPLE_ROWS + 30))
    doc = ingest_document('data.csv', f'Code,Label\n{rows}\n'.encode())
    text = ''.join(_chunks(doc)[0])
    assert 'Value 0' in text and f'Value {SAMPLE_ROWS + 5}' not in text
    assert 'SHEET INDEX' in text and f'first {SAMPLE_ROWS} rows' in text
