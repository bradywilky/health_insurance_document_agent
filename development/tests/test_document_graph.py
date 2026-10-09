"""Graph routing, bounded execution, and per-invocation isolation."""
from backend.agents.document_agent import DOCUMENT_GRAPH, MAX_STEPS, run_document_agent
from backend.preprocessing.documents import ingest_document


def inputs(doc):
    return {'documents': {doc.id: doc}, 'document_ids': [doc.id], 'question': 'Rate?'}


def test_graph_stream_exposes_real_nodes(native_script):
    doc = ingest_document('rates.txt', b'Rate USD 88')
    native_script([{'tool':'read_document','parameters':{'document_id':doc.id}},
                   {'tool':'answer','parameters':{}}, 'USD 88 [E1].'])
    events = list(DOCUMENT_GRAPH.stream(inputs(doc), stream_mode='updates'))
    assert [next(iter(e)) for e in events] == ['init','assess','plan','tools','plan','tools','synthesize']
    assert events[-1]['synthesize']['result']['answer'] == 'USD 88.'


def test_graph_step_limit_synthesizes_without_recursion_error(native_script):
    doc = ingest_document('rates.txt', b'Rate USD 88')
    calls = native_script([{'tool':'read_document','parameters':{'document_id':doc.id}}] * MAX_STEPS
                          + ['USD 88 [E1].'])
    result = run_document_agent({doc.id:doc}, [doc.id], 'Rate?')
    assert len(calls) == MAX_STEPS + 1
    assert len(result['evidence']) == 1
    assert result['protocol']['duplicate_requests'] == MAX_STEPS - 1
    assert any('step limit' in warning for warning in result['limitations'])


def test_graph_invocations_do_not_share_evidence(native_script):
    for name, text in [('a.txt', b'Rate USD 88'), ('b.txt', b'Rate USD 99')]:
        doc = ingest_document(name, text)
        native_script([{'tool':'read_document','parameters':{'document_id':doc.id}},
                       {'tool':'answer','parameters':{}}, 'See rate [E1].'])
        result = run_document_agent({doc.id:doc}, [doc.id], 'Rate?')
        assert len(result['evidence']) == 1
        assert result['evidence'][0]['id'] == 'E1'
        assert result['evidence'][0]['data']['filename'] == name


def test_shared_tools_do_not_import_agents():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / 'backend' / 'tools'
    for path in root.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('backend.agents')
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith('backend.agents') for alias in node.names)
