"""Keeps recent traces in memory for the Streamlit chat app, the command-line tool, evaluations and tests.

backend/observability/tracing.py records every span whether or not Datadog is enabled; enable() adds a listener
that keeps each finished trace here, so a run can be inspected with no Datadog account or network calls.

    capture.enable()
    response = ask_question_local(...)
    summary = capture.summarize(capture.trace(response['trace_id']))

    with capture.recording() as events:      # code that has no trace_id, e.g. run_document_agent
        run_document_agent(...)
"""
from collections import OrderedDict
from contextlib import contextmanager
import os
import threading

from backend.observability import tracing

MAX_TRACES = 200
_traces = OrderedDict()
_recorders = []
_lock = threading.Lock()


def _keep(trace):
    with _lock:
        _traces[trace['trace_id']] = trace['spans']
        _traces.move_to_end(trace['trace_id'])
        while len(_traces) > MAX_TRACES:
            _traces.popitem(last=False)
        for events in _recorders:
            events.extend(trace['spans'])


def enable(content=True):
    """Keep finished traces in memory. content=True also records prompts and responses (TRACE_CONTENT=true).
    Safe to call repeatedly."""
    if content:
        os.environ['TRACE_CONTENT'] = 'true'
    tracing.add_listener(_keep)


def trace(trace_id):
    """Every span event of one trace (one question), oldest first."""
    with _lock:
        return sorted(_traces.get(trace_id, []), key=lambda e: e['start_ns'])


@contextmanager
def recording():
    """Collect the spans of traces that finish while the block runs. Assumes nothing else runs meanwhile."""
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
