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
    """Datadog LLM Observability spans go to the local in-memory capture, as in the development apps."""
    from development.observability import capture
    capture.enable(content=False)
    return capture


@pytest.fixture(autouse=True)
def isolated_audit(monkeypatch, tmp_path):
    """Audit records from tests go to a temporary directory; prompt and response capture stays off.
    Document layout settings from a local .env are cleared so saved paths match the tests."""
    monkeypatch.setenv('AUDIT_STORAGE', 'local')
    monkeypatch.setenv('AUDIT_LOCAL_DIR', str(tmp_path / 'audit'))
    for name in ['AUDIT_CONTENT', 'AUDIT_REQUIRED',
                 'DOCUMENTS_GROUP_NAME', 'DOCUMENTS_S3_PREFIX_BASE', 'DOCUMENTS_S3_PREFIX_PPDOCS']:
        monkeypatch.delenv(name, raising=False)
    # setenv, not delenv, so the value is restored even after an app under test changes it.
    monkeypatch.setenv('TRACE_CONTENT', 'false')
    return tmp_path / 'audit'
