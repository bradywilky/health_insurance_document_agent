import json
import pytest
from development.scripts.make_samples import create_samples


@pytest.fixture(scope='session')
def insurance_workbook(tmp_path_factory):
    return create_samples(tmp_path_factory.mktemp('insurance_samples'))

@pytest.fixture
def native_script(monkeypatch):
    """Script the Bedrock transport, retaining the real LLM wrapper and tool protocol."""
    import copy
    import itertools
    from backend.agents.llm import LLM
    def install(turns):
        replies=iter(turns)
        requests=[]
        ids=itertools.count()
        class Client:
            def converse(self,**kwargs):
                requests.append(copy.deepcopy(kwargs))
                turn=next(replies)
                if isinstance(turn,str):
                    message={'role':'assistant','content':[{'text':turn}]}
                    stop='end_turn'
                else:
                    actions=turn if isinstance(turn,list) else [turn]
                    message={'role':'assistant','content':[{'toolUse':{
                        'toolUseId':f'call-{next(ids)}','name':a['tool'],'input':a['parameters']}} for a in actions]}
                    stop='tool_use'
                return {'output':{'message':message},'stopReason':stop,
                        'usage':{'inputTokens':10,'outputTokens':5}}
        client=Client()
        original=LLM.__init__
        def init(self,*a,**kw):
            original(self,*a,**kw)
            self._client=client
        monkeypatch.setattr(LLM,'__init__',init)
        return requests
    return install


@pytest.fixture(scope='session', autouse=True)
def llm_capture():
    """Finished traces are kept in memory, as in the development apps."""
    from development.observability import capture
    capture.enable(content=False)
    return capture


@pytest.fixture(autouse=True)
def isolated_archive(monkeypatch, tmp_path):
    """Traces from tests are archived under a temporary directory; prompt and response capture stays off.
    Document layout settings from a local .env are cleared so saved paths match the tests."""
    monkeypatch.setenv('TRACE_ARCHIVE', 'local')
    monkeypatch.setenv('TRACE_ARCHIVE_LOCAL_DIR', str(tmp_path / 'archive'))
    for name in ['TRACE_ARCHIVE_REQUIRED',
                 'DOCUMENTS_GROUP_NAME', 'DOCUMENTS_S3_PREFIX_BASE', 'DOCUMENTS_S3_PREFIX_PPDOCS']:
        monkeypatch.delenv(name, raising=False)
    # setenv, not delenv, so the value is restored even after an app under test changes it.
    monkeypatch.setenv('TRACE_CONTENT', 'false')
    # Scripted model replies assume no reasoning step, whatever a local .env sets.
    monkeypatch.setenv('PLANNER_REASONING', 'false')
    # The configured store and archive are reused while settings match; start each test without them.
    from backend.observability import archive
    from backend.storage import s3
    archive._archive.cache_clear()
    s3._store.cache_clear()
    return tmp_path / 'archive'


def archived(directory, name):
    """Archived traces whose root span is `name`, oldest first."""
    files = sorted((directory / name).rglob('*.json'))
    return [json.loads(f.read_text(encoding='utf-8')) for f in files]
