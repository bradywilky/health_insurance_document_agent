"""Local stand-in for Datadog LLM Observability: keeps the span events Datadog would receive, in memory.

Production code only talks to Datadog (backend/observability/llmobs.py). For the Streamlit chat app, the
command-line tool, evaluations and tests, enable() starts ddtrace's LLM Observability locally and replaces its span
writer with an in-memory one, so the same events are captured with no Datadog account, agent or network calls.

    capture.enable()
    response = ask_question_local(...)
    summary = capture.summarize(capture.trace(response['trace_id']))

    with capture.recording() as events:      # code that has no trace_id, e.g. run_document_agent
        run_document_agent(...)

This replaces a private ddtrace attribute (the span writer). requirements.txt pins the ddtrace minor version, and
development/tests/test_observability.py fails if a ddtrace upgrade breaks the swap.
"""
from collections import OrderedDict
from contextlib import contextmanager
import os
import threading

ML_APP = 'health-insurance-document-agent-local'
MAX_TRACES = 200
_traces = OrderedDict()
_recorders = []
_lock = threading.Lock()


class _Writer:
    """Takes the place of ddtrace's LLMObsSpanWriter: keeps events instead of sending them."""

    def enqueue(self, event):
        with _lock:
            _traces.setdefault(event['trace_id'], []).append(event)
            _traces.move_to_end(event['trace_id'])
            while len(_traces) > MAX_TRACES:
                _traces.popitem(last=False)
            for events in _recorders:
                events.append(event)

    def recreate(self):
        return self

    def __getattr__(self, name):
        # start, stop, periodic, flush_queue and the rest of the writer interface have nothing to do here.
        return lambda *args, **kwargs: None


def enable(content=True):
    """Capture LLM Observability events locally. content=True also captures prompts and responses
    (TRACE_CONTENT=true). Safe to call repeatedly."""
    from ddtrace.llmobs import LLMObs
    from ddtrace.trace import tracer
    if content:
        os.environ['TRACE_CONTENT'] = 'true'
    if not LLMObs.enabled:
        tracer.enabled = False  # nothing is sent to a Datadog agent locally
        LLMObs.enable(agent_service=ML_APP, agentless_enabled=False, integrations_enabled=False)
    if not LLMObs.enabled:
        raise RuntimeError('LLM Observability could not be enabled (is DD_LLMOBS_ENABLED=false?)')
    writer = _Writer()
    LLMObs._instance._llmobs_span_writer = writer
    LLMObs._instance.tracer._span_aggregator.llmobs_processor._llmobs_span_writer = writer


def trace(trace_id):
    """Every captured span event of one trace (one question), oldest first."""
    with _lock:
        return sorted(_traces.get(trace_id, []), key=lambda e: e['start_ns'])


@contextmanager
def recording():
    """Collect the events captured while the block runs. Assumes nothing else runs questions meanwhile."""
    events = []
    with _lock:
        _recorders.append(events)
    try:
        yield events
    finally:
        with _lock:
            _recorders.remove(events)


def kind(event):
    return event['meta']['span']['kind']


def summarize(events):
    """Per-question view of the captured events: totals, then each model call and tool execution in order.

    Times are milliseconds from the first span; gap_ms is the time since the previous model call ended."""
    if not events:
        return {'totals': {'llm_calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
                           'largest_call': None, 'duration_ms': 0, 'model_ms': 0, 'models': []},
                'calls': [], 'tools': []}
    events = sorted(events, key=lambda e: e['start_ns'])
    first = min(e['start_ns'] for e in events)
    end = max(e['start_ns'] + e['duration'] for e in events)
    calls, previous_end = [], None
    for number, e in enumerate((e for e in events if kind(e) == 'llm'), 1):
        meta, metrics = e['meta'], e.get('metrics') or {}
        metadata = meta.get('metadata') or {}
        calls.append({
            'call': number, 'step': e['name'], 'model': meta.get('model_name'),
            'start_ms': _ms(e['start_ns'] - first), 'duration_ms': _ms(e['duration']),
            'gap_ms': _ms(e['start_ns'] - previous_end) if previous_end is not None else None,
            'input_tokens': metrics.get('input_tokens', 0), 'output_tokens': metrics.get('output_tokens', 0),
            'total_tokens': metrics.get('total_tokens', 0), 'stop_reason': metadata.get('stop_reason'),
            'tool_calls': metadata.get('tool_calls') or [], 'status': e['status'],
            'input': (meta.get('input') or {}).get('messages') or [],
            'output': (meta.get('output') or {}).get('messages') or []})
        previous_end = e['start_ns'] + e['duration']
    tools = [{'tool': e['name'], 'start_ms': _ms(e['start_ns'] - first), 'duration_ms': _ms(e['duration']),
              'status': e['status'], **{k: v for k, v in (e['meta'].get('metadata') or {}).items()
                                        if k in {'evidence_id', 'operation', 'matches', 'blocks', 'matched_rows'}}}
             for e in events if kind(e) == 'tool']
    largest = max(calls, key=lambda c: c['total_tokens'], default=None)
    return {'totals': {
                'llm_calls': len(calls), 'input_tokens': sum(c['input_tokens'] for c in calls),
                'output_tokens': sum(c['output_tokens'] for c in calls),
                'total_tokens': sum(c['total_tokens'] for c in calls),
                'largest_call': largest['call'] if largest else None,
                'duration_ms': _ms(end - first), 'model_ms': round(sum(c['duration_ms'] for c in calls), 1),
                'models': sorted({c['model'] for c in calls if c['model']})},
            'calls': calls, 'tools': tools}


def _ms(nanoseconds):
    return round(nanoseconds / 1e6, 1)
