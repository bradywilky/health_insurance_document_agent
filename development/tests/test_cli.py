import json
import sys
import pytest
from development.cli.run_local import main
from backend.storage.local import prepare_local_file, prepare_preprocessed_directory
from backend.preprocessing.documents import document_from_table_inputs
from backend.preprocessing.tables import preprocess_file


def script(doc, native_script):
    return native_script([
        {'tool':'table_tool','parameters':{'document_id':doc.id,'name':'query_table',
         'parameters':{'sheet_name':'Sheet1','aggregations':[{'column':'Revenue','op':'sum','as':'total'}]}}},
        {'tool':'answer','parameters':{}}, 'Revenue total 450 [E1].'])


@pytest.mark.parametrize('stream', [False, True])
def test_cli_uses_document_graph_with_csv_options(tmp_path, monkeypatch, native_script, capsys, stream):
    path = tmp_path / 'sales.csv'
    path.write_bytes('Region;Revenue\nOuest;100\nEst;200\nOuest;150\n'.encode('cp1252'))
    doc = document_from_table_inputs(**prepare_local_file(path, encoding='cp1252', delimiter=';'))
    requests = script(doc, native_script)
    args = ['run_local',str(path),'Total?', '--encoding','cp1252','--delimiter',';', '--model','maverick']
    if stream:
        args.append('--stream')
    monkeypatch.setattr(sys,'argv',args)
    main()
    output = capsys.readouterr().out
    assert '450' in output and 'LLM calls: 3' in output
    assert len(requests) == 3
    if stream:
        assert '"node": "plan"' in output
        assert 's3_client' not in output and '"evidence"' not in output


def test_cli_exported_directory_inspect_and_answer(tmp_path, monkeypatch, native_script, capsys):
    path = tmp_path / 'sales.csv'
    path.write_text('Revenue\n450\n')
    objects, _ = preprocess_file(path)
    folder = tmp_path / 'export'
    folder.mkdir()
    for name, body in objects.items():
        (folder / name).write_bytes(body)
    args = ['run_local',str(folder),'Total?', '--preprocessed']
    monkeypatch.setattr(sys,'argv',args + ['--inspect'])
    main()
    assert 'Sheet1' in capsys.readouterr().out
    doc = document_from_table_inputs(**prepare_preprocessed_directory(folder))
    script(doc, native_script)
    monkeypatch.setattr(sys,'argv',args)
    main()
    assert '450' in capsys.readouterr().out


def test_table_evaluation_uses_document_graph(tmp_path, monkeypatch, native_script):
    from development.evaluation.tables import main as evaluate
    path = tmp_path / 'sales.csv'
    path.write_text('Revenue\n450\n')
    doc = document_from_table_inputs(**prepare_local_file(path))
    requests = script(doc, native_script)
    questions = tmp_path / 'questions.json'
    questions.write_text(json.dumps([{'id':'total','question':'Total?',
                                     'expected':'SECRET_EXPECTATION', 'sheets':['Sheet1']}]))
    output = tmp_path / 'results.jsonl'
    monkeypatch.setattr(sys,'argv',['evaluate','--file',str(path),'--questions',str(questions),'--output',str(output)])
    monkeypatch.delenv('BEDROCK_MODEL_ID', raising=False)
    evaluate()
    record = json.loads(output.read_text())
    assert record['status'] == 'needs_human_review'
    assert record['evidence'][0]['tool'] == 'table_tool'
    assert 'SECRET_EXPECTATION' not in json.dumps(requests)
