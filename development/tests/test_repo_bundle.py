import json
import pytest
from transfer.restore_repo import MAGIC, SEPARATOR, END, digest, parse_bundle, restore_bundle


def bundle(name="backend/example.py", content="print('hello')\n"):
    return MAGIC + SEPARATOR + json.dumps({"path":name,"characters":len(content),"sha256":digest(content)}) + "\n" + content + "\n" + END


def test_restore_unicode_empty_and_separator_content(tmp_path):
    content = "cafe \u00e9\n" + SEPARATOR + END
    source = tmp_path / 'bundle.txt'
    source.write_bytes(('\ufeff'+bundle(content=content)).replace('\n','\r\n').encode())
    target = tmp_path / 'restored'
    assert restore_bundle(source, target) == 1
    assert (target/'backend/example.py').read_text(encoding='utf-8') == content
    assert parse_bundle(bundle(content='')) == [('backend/example.py', '')]


@pytest.mark.parametrize('name',['../escape.py','/absolute.py','C:/escape.py','a/../../b','.env','.git/config','runs/trace.json','AUX.txt'])
def test_reject_unsafe_paths(name):
    with pytest.raises(ValueError):
        parse_bundle(bundle(name=name))


def test_corruption_rejected_before_writes(tmp_path):
    source = tmp_path/'bundle.txt'
    source.write_text(bundle().replace("hello", "jello"))
    target = tmp_path/'restored'
    with pytest.raises(ValueError):
        restore_bundle(source,target)
    assert not target.exists()


def test_refuses_existing_destination(tmp_path):
    source=tmp_path/'bundle.txt'
    source.write_text(bundle())
    with pytest.raises(ValueError,match='already exists'):
        restore_bundle(source,tmp_path)


def test_export_excludes_bundle_and_is_repeatable(tmp_path):
    import subprocess
    from transfer.create_bundle import export_bundle
    subprocess.run(['git','init',str(tmp_path)], check=True, capture_output=True)
    (tmp_path/'transfer').mkdir()
    (tmp_path/'example.py').write_text('print("hello")\n')
    output=tmp_path/'transfer/repository-bundle.txt'
    assert export_bundle(tmp_path,output)==1
    original=output.read_bytes()
    assert export_bundle(tmp_path,output)==1
    assert output.read_bytes()==original
    alternate=tmp_path/'other.txt'
    assert export_bundle(tmp_path,alternate)==1
    assert alternate.read_bytes()==original
