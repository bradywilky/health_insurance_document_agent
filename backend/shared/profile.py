"""Planner-facing view of a stored semantic profile (see backend/agents/enrichment.py).

Kept small because it rides along in every planner call: the document card, references, and only glossary
meanings the document states or the uploader confirmed. Model guesses and column roles stay in the stored
profile and the upload portal, where people can review them.
"""


def profile_hints(profile, *, max_glossary=15):
    order = {'confirmed': 0, 'defined': 1}
    glossary = sorted((g for g in profile.get('glossary', []) if g['source'] in order),
                      key=lambda g: order[g['source']])[:max_glossary]
    view = {'note': 'Generated at upload by an LLM. Hints only, not evidence: verify in the source and cite source '
                    'blocks or rows. "confirmed" meanings were reviewed by the uploader.',
            'document_type': profile.get('document_type'), 'status': profile.get('status'),
            'summary': profile.get('summary', '')[:300],
            'effective_dates': [f"{d['label']}: {d['value']}" for d in profile.get('effective_dates', [])],
            'identifiers': [i['identifier'] for i in profile.get('identifiers', [])],
            'references': [f"{r['relationship']} {r['target']}" for r in profile.get('references', [])],
            'glossary': [{'term': g['term'], 'source': g['source'],
                          # A defined term shows the document's own words, not the model's paraphrase.
                          'meaning': g['quote'] if g['source'] == 'defined' and g.get('quote') else g['meaning']}
                         for g in glossary],
            'status_values': {s['value']: s['meaning'] for s in profile.get('status_values', [])}}
    if not profile.get('meta', {}).get('coverage', {}).get('complete', True):
        view['coverage'] = 'Profile covers only the beginning of a long document.'
    return {k: v for k, v in view.items() if v not in ('', [], {}, None)}
