"""Datadog LLM Observability spans for questions, graph steps, Bedrock calls and tool executions.

In AWS Lambda the Datadog layers (Datadog-Python3xx and Datadog-Extension) turn LLM Observability on with
DD_LLMOBS_ENABLED=true and ship the spans to Datadog; see the README. When LLM Observability is not enabled every
helper here is a no-op, so the application runs unchanged without Datadog.

agent     document_agent                 one per question: request and session IDs, status
workflow  init | assess | plan | tools | synthesize, and ingest_document in the upload portal
llm       one per Bedrock call           model, step, tokens, stop reason, tool calls requested
tool      one per tool execution         evidence ID, document IDs, match and row counts

TRACE_CONTENT=true  also send prompts, responses, the question, the answer and tool parameters. Off by default:
                    they can contain PHI.
"""
from contextlib import contextmanager
import functools
import json
import os

from ddtrace.llmobs import LLMObs

MODEL_PROVIDER = 'amazon_bedrock'


def content_capture():
    return os.getenv('TRACE_CONTENT', 'false').strip().lower() == 'true'


@contextmanager
def span(kind, name, **options):
    """Start an LLM Observability span of `kind` (agent, workflow, llm, tool); yields None when disabled.

    Exceptions are recorded on the span and re-raised."""
    if not LLMObs.enabled:
        yield None
        return
    with getattr(LLMObs, kind)(name=name, **options) as current:
        yield current


def annotate(current, *, metadata=None, metrics=None, tags=None):
    """Attach metadata, numeric metrics and tags; None values are dropped."""
    if current is None:
        return
    fields = {'metadata': _drop_none(metadata), 'metrics': _drop_none(metrics), 'tags': _drop_none(tags)}
    LLMObs.annotate(span=current, **{k: v for k, v in fields.items() if v})


def annotate_content(current, *, input_data=None, output_data=None):
    """Attach input and output text only when TRACE_CONTENT=true."""
    if current is None or not content_capture():
        return
    fields = {'input_data': input_data, 'output_data': output_data}
    LLMObs.annotate(span=current, **{k: v for k, v in fields.items() if v is not None})


def mark_error(current, message):
    """Mark a span as failed without raising, e.g. a tool that returned an error result."""
    if current is not None:
        current.error = 1
        current.set_tag('error.message', str(message)[:500])


def trace_id_of(current):
    """LLM Observability trace ID, for correlating audit records with Datadog; None when disabled."""
    if current is None or not LLMObs.enabled:
        return None
    exported = LLMObs.export_span(current)
    return exported['trace_id'] if exported else None


def current_trace_id():
    """Trace ID of the active span (e.g. ingest_document in the upload portal), or None."""
    if not LLMObs.enabled:
        return None
    current = LLMObs._instance._current_span()
    return trace_id_of(current) if current is not None else None


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


def _drop_none(values):
    return {k: v for k, v in (values or {}).items() if v is not None}
