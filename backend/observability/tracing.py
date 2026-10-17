"""Traces for questions, graph steps, Bedrock calls, tool executions and document changes.

Every span is recorded here and, when Datadog LLM Observability is enabled, also sent to Datadog with the same
name, metadata, metrics, tags and content. When an entry point's root span ends, the whole trace is written to the
trace archive (backend/observability/archive.py): the audit record, readable without Datadog.

In AWS Lambda the Datadog layers (Datadog-Python3xx and Datadog-Extension) turn LLM Observability on with
DD_LLMOBS_ENABLED=true; see the README. Without Datadog, traces are still recorded and archived.

agent     document_agent                 one per question: request and session IDs, versions, status, totals
workflow  init | assess | plan | tools | synthesize
          ingest_document | delete_document | save_profile   one per document change
llm       one per Bedrock call           model, step, tokens, stop reason, tool calls requested
embedding one per Bedrock embedding call model, dimensions, tokens (passages at upload, semantic_search queries)
tool      one per tool execution         evidence ID, document IDs, match and row counts

TRACE_CONTENT=true  also record prompts, responses, the question, the answer, and tool parameters and results.
                    Off by default: they can contain PHI. It applies to Datadog and the archive alike.

Span events have the shape of Datadog's LLM Observability span events, so a trace reads the same in either place.
"""
from contextlib import contextmanager, nullcontext
import contextvars
import functools
import json
import logging
import os
import threading
import time
import uuid

from ddtrace.llmobs import LLMObs

MODEL_PROVIDER = 'amazon_bedrock'
logger = logging.getLogger(__name__)
_current = contextvars.ContextVar('current_span', default=None)
_listeners = []


def content_capture():
    return os.getenv('TRACE_CONTENT', 'false').strip().lower() == 'true'


def add_listener(listener):
    """Call listener(trace) with every finished trace (see Trace.record); used by the local development capture."""
    if listener not in _listeners:
        _listeners.append(listener)


class Trace:
    def __init__(self, trace_id):
        self.trace_id = trace_id
        self.events = []
        self.lock = threading.Lock()

    def record(self, root):
        with self.lock:
            spans = sorted(self.events, key=lambda e: e['start_ns'])
        return {'trace_id': self.trace_id, 'name': root.name, 'kind': root.kind,
                'started_at': _iso(root.start_ns), 'duration_ms': round(root.duration_ns / 1e6, 1),
                'status': 'error' if root.error else 'ok', 'spans': spans}


class Span:
    """One span. Annotate it with the helpers below; they also forward to Datadog when it is enabled."""

    def __init__(self, kind, name, parent, datadog, options):
        exported = LLMObs.export_span(datadog) if datadog is not None else None
        self.kind, self.name, self.parent, self.datadog = kind, name, parent, datadog
        self.span_id = (exported or {}).get('span_id') or uuid.uuid4().hex[:16]
        self.trace = parent.trace if parent else Trace((exported or {}).get('trace_id') or uuid.uuid4().hex)
        self.session_id = options.get('session_id') or (parent.session_id if parent else None)
        self.model = {k: options[k] for k in ('model_name', 'model_provider') if options.get(k)}
        self.metadata, self.metrics, self.tags = {}, {}, {}
        self.input = self.output = self.error = None
        self.start_ns, self._start = time.time_ns(), time.perf_counter_ns()
        self.duration_ns = 0

    def finish(self):
        self.duration_ns = time.perf_counter_ns() - self._start
        meta = {'span': {'kind': self.kind}, **self.model, 'metadata': self.metadata}
        if self.input is not None:
            meta['input'] = self.input
        if self.output is not None:
            meta['output'] = self.output
        if self.error:
            meta['error.message'] = self.error
        event = {'trace_id': self.trace.trace_id, 'span_id': self.span_id,
                 'parent_id': self.parent.span_id if self.parent else 'undefined', 'name': self.name,
                 'start_ns': self.start_ns, 'duration': self.duration_ns,
                 'status': 'error' if self.error else 'ok', 'meta': meta, 'metrics': self.metrics,
                 'tags': [f'{k}:{v}' for k, v in self.tags.items()]}
        if self.session_id:
            event['session_id'] = self.session_id
        with self.trace.lock:
            self.trace.events.append(event)


@contextmanager
def span(kind, name, *, archive=None, **options):
    """Start a span of `kind` (agent, workflow, llm, embedding, tool), a child of the span open in this context.

    archive: on a root span, the trace archive that receives the finished trace (archive.configured_archive()).
    Exceptions are recorded on the span and re-raised."""
    parent = _current.get()
    with getattr(LLMObs, kind)(name=name, **options) if LLMObs.enabled else nullcontext() as datadog:
        current = Span(kind, name, parent, datadog, options)
        token = _current.set(current)
        try:
            yield current
        except BaseException as exc:
            current.error = current.error or f'{type(exc).__name__}: {exc}'[:500]
            raise
        finally:
            _current.reset(token)
            current.finish()
            if parent is None:
                _finished(current, archive)


def _finished(root, archive):
    record = root.trace.record(root)
    for listener in list(_listeners):
        try:
            listener(record)
        except Exception:
            logger.exception('Trace listener failed')
    if archive is not None:
        from backend.observability.archive import write
        write(archive, record)


def annotate(current, *, metadata=None, metrics=None, tags=None):
    """Attach metadata, numeric metrics and tags; None values are dropped."""
    if current is None:
        return
    fields = {'metadata': _drop_none(metadata), 'metrics': _drop_none(metrics), 'tags': _drop_none(tags)}
    current.metadata.update(fields['metadata'])
    current.metrics.update(fields['metrics'])
    current.tags.update(fields['tags'])
    if current.datadog is not None:
        LLMObs.annotate(span=current.datadog, **{k: v for k, v in fields.items() if v})


def annotate_content(current, *, input_data=None, output_data=None):
    """Attach input and output text only when TRACE_CONTENT=true."""
    if current is None or not content_capture():
        return
    if input_data is not None:
        current.input = _content(current, input_data)
    if output_data is not None:
        current.output = _content(current, output_data)
    if current.datadog is not None:
        fields = {'input_data': input_data, 'output_data': output_data}
        LLMObs.annotate(span=current.datadog, **{k: v for k, v in fields.items() if v is not None})


def _content(current, data):
    if current.kind == 'llm':
        return {'messages': data}
    return {'value': data if isinstance(data, str) else json.dumps(data, default=str)}


def mark_error(current, message):
    """Mark a span as failed without raising, e.g. a tool that returned an error result."""
    if current is None:
        return
    current.error = str(message)[:500]
    if current.datadog is not None:
        current.datadog.error = 1
        current.datadog.set_tag('error.message', current.error)


def trace_id_of(current):
    """Trace ID: Datadog's when LLM Observability is enabled, so the archived trace and Datadog's match."""
    return current.trace.trace_id if current is not None else None


def traced_node(name):
    """Wrap a LangGraph node so each step is its own workflow span."""
    def decorate(function):
        @functools.wraps(function)
        def wrapper(state):
            with span('workflow', name) as current:
                annotate(current, metadata={'step': state.get('steps')})
                update = function(state)
                if isinstance(update, dict):
                    annotate(current, metadata={'finished': update.get('finished'),
                                                'returned_result': 'result' in update})
                return update
        return wrapper
    return decorate


def bedrock_messages(messages, system=None):
    """Bedrock Converse messages in LLM Observability form: text, tool calls and tool results."""
    out = [{'role': 'system', 'content': system}] if system else []
    for message in messages or []:
        blocks = message.get('content') or []
        item = {'role': message.get('role', ''),
                'content': '\n'.join(b['text'] for b in blocks if 'text' in b)}
        calls = [{'name': b['toolUse']['name'], 'arguments': b['toolUse'].get('input') or {},
                  'tool_id': b['toolUse'].get('toolUseId', ''), 'type': 'toolUse'}
                 for b in blocks if 'toolUse' in b]
        results = [{'result': '\n'.join(c['text'] if 'text' in c else json.dumps(c.get('json'), default=str)
                                        for c in b['toolResult'].get('content', [])),
                    'tool_id': b['toolResult'].get('toolUseId', ''), 'type': 'toolResult'}
                   for b in blocks if 'toolResult' in b]
        if calls:
            item['tool_calls'] = calls
        if results:
            item['tool_results'] = results
        out.append(item)
    return out


def _iso(ns):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).isoformat(timespec='milliseconds')


def _drop_none(values):
    return {k: v for k, v in (values or {}).items() if v is not None}
