"""Single entry point for document question answering.

ask_question is the production entry point (the Lambda handler calls it with a group and saved filenames).
ask_question_local is for development callers (the Streamlit app, the command-line tool, tests): it takes
already-loaded documents and adds a progress hook. Both validate the request, load the documents, run
the agent, archive the trace, and return the same JSON-serializable response, whose `status`
tells the caller what happened:

answered              `answer` is the answer; `sources` and `limitations` support it
needs_clarification   `answer` is a clarifying question; resend the same question with `clarification` set to
                      one of `clarification.options` or the user's own wording
no_evidence           nothing relevant was retrieved; `message` explains
invalid_request       bad input (no question, too many documents, unknown model, ...); `message` explains
documents_unavailable a saved document could not be loaded
model_unavailable     Bedrock throttled, timed out or was unreachable after retries
error                 anything else; `error` holds the exception type and message

STATUS_CODES maps each status to an HTTP-style code for API callers.
"""
import logging
import uuid

from backend.agents import ambiguity as amb
from backend.agents import document_agent
from backend.agents.llm import Usage
from backend.config.settings import MODELS
from backend.observability import tracing
from backend.storage.documents import check_group_name
from backend.storage.s3 import configured_store
from backend.observability.archive import (code_version, configured_archive, digest, document_summary,
                                           evidence_summary)

MAX_DOCUMENTS = 10
MAX_QUESTION_CHARS = 4000
STATUS_CODES = {'answered': 200, 'needs_clarification': 200, 'no_evidence': 204, 'invalid_request': 400,
                'documents_unavailable': 404, 'model_unavailable': 503, 'error': 500}
MESSAGES = {
    'no_evidence': 'I could not retrieve supporting content from the selected files. '
                   'Try a different question or inspect extraction warnings.',
    'model_unavailable': 'The model service is busy or unreachable. Try again shortly.',
}
_UNAVAILABLE = {'ThrottlingException', 'ServiceUnavailableException', 'ModelNotReadyException',
                'ModelTimeoutException', 'ReadTimeoutError', 'ConnectTimeoutError', 'EndpointConnectionError'}
logger = logging.getLogger(__name__)


class RequestError(ValueError):
    """Invalid request or unavailable documents; reported to the caller rather than raised."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def ask_question(question, *, document_filenames, group_name=None, history=None, session_id=None, user_id=None,
                 clarification=None, model=None, ambiguity='off', app_id='api'):
    """Production entry point: answer `question` from saved documents. Never raises; see the module docstring.

    document_filenames  saved filenames, e.g. 'rates.txt' (1 to MAX_DOCUMENTS), read from
                   <DOCUMENTS_S3_PREFIX_BASE><group_name>/<DOCUMENTS_S3_PREFIX_PPDOCS>/<filename>/
    group_name     the group folder, e.g. 'aol.com'; default DOCUMENTS_GROUP_NAME. Required for S3 storage.
    history        prior turns: [{'role': 'user' | 'assistant', 'content': str}]
    session_id     conversation identifier; Datadog groups the conversation's traces by it
    user_id        who asked, e.g. from the signed-in user's profile; a tag on the trace
    clarification  the reading the user chose after a needs_clarification response
    model          a key of backend.config.settings.MODELS; default from TABLES_MODEL
    ambiguity      'off' (answer directly), 'assumptions' (state the reading used) or 'ask'
    app_id         which application called; a tag on the trace

    Documents are read from the store the environment configures (backend/storage/s3.py), and the trace is saved
    to the configured archive (backend/observability/archive.py); both are reused across requests.
    """
    return _run(question, selected=document_filenames, load=lambda: _load(document_filenames, group_name),
                history=history, session_id=session_id, user_id=user_id, clarification=clarification, model=model,
                ambiguity=ambiguity, app_id=app_id, group=group_name)


def ask_question_local(question, *, documents, history=None, session_id=None, user_id=None, clarification=None,
                       model=None, ambiguity='off', app_id='local', on_step=None):
    """Development entry point for the Streamlit app, the command-line tool and tests. Same behavior and response
    as ask_question, but takes already-loaded Document objects and adds development hooks:

    documents      Document objects (e.g. a local file that was never saved to the document store)
    on_step        callback after each research step, for a progress display
    """
    return _run(question, selected=documents, load=lambda: list(documents), history=history,
                session_id=session_id, user_id=user_id, clarification=clarification, model=model,
                ambiguity=ambiguity, app_id=app_id, on_step=on_step)


def _run(question, *, selected, load, history, session_id, user_id, clarification, model, ambiguity, app_id,
         group=None, on_step=None):
    """Validate, load and run the agent inside one trace, which is archived when it ends."""
    usage = Usage()
    loaded, result, error, status = [], None, None, 'error'
    with tracing.span('agent', 'document_agent', session_id=session_id,
                      archive=configured_archive()) as root:
        request_id = uuid.uuid4().hex
        text = question if isinstance(question, str) else ''
        tracing.annotate(root, metadata={
            'request_id': request_id, 'ambiguity_mode': ambiguity, 'model': model, 'group_name': group,
            'code_version': code_version(), 'prompt_version': document_agent.PROMPT_VERSION,
            'question_sha256': digest(text), 'question_chars': len(text), 'clarified': bool(clarification)},
            tags={'app_id': app_id, 'user_id': user_id})
        tracing.annotate_content(root, input_data={'question': question, 'clarification': clarification})
        try:
            question, history, clarification = _validate(question, history, clarification, model, ambiguity,
                                                         selected)
            loaded = load()
            tracing.annotate(root, metadata={'documents': [document_summary(d) for d in loaded]})
            # Module attribute, so tests can replace the agent.
            result = document_agent.run_document_agent(
                {d.id: d for d in loaded}, [d.id for d in loaded], question, history=history, usage=usage,
                on_step=on_step, model=model, ambiguity=ambiguity, clarification=clarification)
            status = result.get('status') or 'answered'
        except RequestError as exc:
            status, error = exc.status, str(exc)
        except Exception as exc:
            status, error = classify(exc), f'{type(exc).__name__}: {exc}'[:1000]
            logger.exception('Question %s failed', request_id)
        result = result or {}
        # The ambiguity assessment quotes readings of the question, so it is content, not metadata.
        protocol = {k: v for k, v in (result.get('protocol') or {}).items() if k != 'ambiguity_assessment'}
        tracing.annotate(root, metadata={
            'status': status, 'error': error, 'totals': usage.totals(),
            'answer_sha256': digest(result['answer']) if result.get('answer') else None,
            'protocol': protocol or None, 'coverage': result.get('coverage'),
            'limitations_count': len(result.get('limitations') or []),
            'evidence': evidence_summary(result.get('evidence'))}, tags={'status': status})
        tracing.annotate_content(root, output_data=result.get('answer'))
        if status not in {'answered', 'needs_clarification', 'no_evidence'}:
            tracing.mark_error(root, error or status)
        response = _response(status, result, error, loaded, request_id, tracing.trace_id_of(root), session_id,
                             question, usage)
    return response


def classify(exc):
    """Map an exception to a status. Botocore errors carry their AWS error code in exc.response."""
    names = {type(exc).__name__, str((getattr(exc, 'response', None) or {}).get('Error', {}).get('Code', ''))}
    if names & _UNAVAILABLE:
        return 'model_unavailable'
    return 'error'


def _validate(question, history, clarification, model, ambiguity, selected):
    if not isinstance(question, str) or not question.strip():
        raise RequestError('invalid_request', 'A question is required.')
    if len(question) > MAX_QUESTION_CHARS:
        raise RequestError('invalid_request', f'Questions are limited to {MAX_QUESTION_CHARS} characters.')
    if model is not None and model not in MODELS:
        raise RequestError('invalid_request', f'model must be one of {list(MODELS)}.')
    if ambiguity not in amb.MODES:
        raise RequestError('invalid_request', f'ambiguity must be one of {list(amb.MODES)}.')
    if not isinstance(selected, (list, tuple)) or not selected:
        raise RequestError('invalid_request', 'Select at least one document.')
    if len(selected) > MAX_DOCUMENTS:
        raise RequestError('invalid_request', f'Select at most {MAX_DOCUMENTS} documents.')
    if clarification is not None and (not isinstance(clarification, str) or not clarification.strip()):
        raise RequestError('invalid_request', 'clarification must be nonempty text when provided.')
    turns = []
    for turn in history or []:
        if not isinstance(turn, dict) or turn.get('role') not in {'user', 'assistant'} \
                or not isinstance(turn.get('content'), str):
            raise RequestError('invalid_request', "history items must be {'role': 'user'|'assistant', 'content': str}.")
        turns.append({'role': turn['role'], 'content': turn['content']})
    return question.strip(), turns, clarification.strip() if clarification else None


def _load(filenames, group_name):
    if not all(isinstance(f, str) and f for f in filenames):
        raise RequestError('invalid_request', 'document_filenames must be nonempty strings.')
    store = configured_store(read_only=True)
    if group_name is not None:
        try:
            store = store.for_group(check_group_name(group_name))
        except ValueError as exc:
            raise RequestError('invalid_request', str(exc)) from exc
    if store.requires_group and store.group is None:
        raise RequestError('invalid_request', 'group_name is required (or set DOCUMENTS_GROUP_NAME).')
    loaded = []
    for name in dict.fromkeys(filenames):
        try:
            loaded.append(store.load(name))
        except Exception as exc:
            if classify(exc) != 'error':
                raise
            raise RequestError('documents_unavailable', f'Could not open saved document {name}: {exc}') from exc
    return loaded


def _location(location):
    if isinstance(location, dict):
        return ', '.join(f'{k} {v}' for k, v in location.items() if v != '')
    return str(location or '')


def sources(evidence):
    """Evidence reshaped for display: passages with file and location, table results, calculations."""
    out = []
    for record in evidence or []:
        data = record.get('data') or {}
        passages = [{'filename': p.get('filename', data.get('filename', '')), 'location': _location(p.get('location')),
                     'text': p.get('text', '')}
                    for p in data.get('matches', data.get('blocks', [])) if isinstance(p, dict)]
        out.append({'id': record.get('id'), 'tool': record.get('tool'), 'auto': bool(record.get('auto')),
                    'filename': data.get('filename'), 'passages': passages, 'table': data.get('table_result'),
                    'calculation': data if record.get('tool') in {'calculate', 'date_calculate'} else None,
                    'error': data.get('error'),
                    'partial': bool(data.get('truncated') or data.get('next_offset') is not None)})
    return out


def _response(status, result, error, loaded, request_id, trace_id, session_id, question, usage):
    result = result or {}
    interpretation = result.get('interpretation')
    if interpretation and interpretation.get('alternatives'):
        interpretation = {**interpretation, 'note': 'Answered as: ' + interpretation['used']}
    clarification = result.get('clarification')
    if status == 'needs_clarification' and clarification:
        clarification = {**clarification, 'original_question': question}
    answer = result.get('answer') if status in {'answered', 'needs_clarification'} else None
    message = MESSAGES.get(status) or (error if status != 'answered' else None)
    if status == 'error':
        message = f'The question could not be completed: {error}'
    return {
        'status': status, 'status_code': STATUS_CODES[status], 'answer': answer, 'message': message,
        'request_id': request_id, 'trace_id': trace_id, 'session_id': session_id,
        'documents': [{'id': d.id, 'name': d.name, 'kind': d.kind,
                       'key': (getattr(d, 'storage_ref', None) or {}).get('manifest_key'),
                       'warnings': list(getattr(d, 'warnings', []) or [])} for d in loaded],
        'sources': sources(result.get('evidence')), 'evidence': result.get('evidence', []),
        'limitations': result.get('limitations', []), 'coverage': result.get('coverage'),
        'interpretation': interpretation, 'clarification': clarification if status == 'needs_clarification' else None,
        'protocol': result.get('protocol'), 'usage': usage.totals(),
        'error': {'type': error.split(':', 1)[0], 'message': error} if error and status not in {
            'invalid_request', 'documents_unavailable'} else None,
    }
