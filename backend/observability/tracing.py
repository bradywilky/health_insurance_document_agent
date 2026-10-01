"""OpenTelemetry spans for questions, graph steps, Bedrock calls and tool executions.

Spans follow the OpenTelemetry GenAI conventions (gen_ai.* attributes), so CloudWatch GenAI Observability
(AgentCore Observability) and other OTel backends can read them. Without a configured tracer provider the
OpenTelemetry API is a no-op, so this module costs nothing until tracing is turned on.

TRACING=off (default)  no provider here; an outer launcher such as ADOT's opentelemetry-instrument may add one
TRACING=file           development: write spans as JSON lines to TRACING_FILE (default data/traces/spans.jsonl)
TRACING=console        development: print spans
TRACE_CONTENT=true     also attach prompt/response text to spans. Off by default: it can contain PHI.
"""
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import threading

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

TRACER_NAME = 'health_insurance_document_agent'
CONTENT_LIMIT = 4000
_configured = False
_lock = threading.Lock()


def tracer():
    return trace.get_tracer(TRACER_NAME)


def content_capture():
    return os.getenv('TRACE_CONTENT', 'false').strip().lower() == 'true'


def _clean(attributes):
    """OTel attributes must be scalars or lists of scalars; drop empties."""
    clean = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            clean[key] = [str(v) for v in value]
        elif isinstance(value, (str, bool, int, float)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


@contextmanager
def span(name, **attributes):
    """Start a child of the current span; exceptions are recorded and mark the span as an error."""
    with tracer().start_as_current_span(name, attributes=_clean(attributes)) as current:
        yield current


def set_attributes(current, **attributes):
    if current is not None and current.is_recording():
        current.set_attributes(_clean(attributes))


def mark_error(current, message):
    if current is not None and current.is_recording():
        current.set_status(Status(StatusCode.ERROR, str(message)[:500]))


def add_content(current, name, **fields):
    """Attach text only when TRACE_CONTENT=true."""
    if content_capture() and current is not None and current.is_recording():
        current.add_event(name, _clean({k: json.dumps(v, default=str)[:CONTENT_LIMIT] if not isinstance(v, str)
                                        else v[:CONTENT_LIMIT] for k, v in fields.items()}))


def trace_id_of(current):
    """Hex trace id for correlating audit records with traces; None when tracing is off."""
    context = current.get_span_context() if current is not None else None
    return format(context.trace_id, '032x') if context and context.is_valid else None


def traced_node(name):
    """Wrap a LangGraph node so each step is its own span."""
    def decorate(function):
        @functools.wraps(function)
        def wrapper(state):
            with span(f'graph.node {name}', **{'app.graph.node': name, 'app.step': state.get('steps')}) as current:
                update = function(state)
                if isinstance(update, dict):
                    set_attributes(current, **{'app.graph.finished': update.get('finished'),
                                               'app.graph.returned_result': 'result' in update})
                return update
        return wrapper
    return decorate


class _JsonLinesExporter:
    """Minimal span exporter for local development: one JSON object per span."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult
        with _lock, self.path.open('a', encoding='utf-8') as out:
            for item in spans:
                out.write(item.to_json(indent=None) + '\n')
        return SpanExportResult.SUCCESS

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis=30000):
        return True


def configure_tracing():
    """Install a development exporter when TRACING asks for one. Safe to call repeatedly."""
    global _configured
    mode = os.getenv('TRACING', 'off').strip().lower()
    if _configured or mode == 'off':
        return
    if mode not in {'file', 'console'}:
        raise ValueError('TRACING must be off, file or console')
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    with _lock:
        if _configured:
            return
        existing = trace.get_tracer_provider()
        if isinstance(existing, TracerProvider):
            _configured = True  # an outer launcher (e.g. ADOT) already installed a provider
            return
        provider = TracerProvider(resource=Resource.create({'service.name': os.getenv('OTEL_SERVICE_NAME', TRACER_NAME)}))
        from backend.observability.audit import project_path
        exporter = (_JsonLinesExporter(project_path(os.getenv('TRACING_FILE', 'data/traces/spans.jsonl')))
                    if mode == 'file' else ConsoleSpanExporter())
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _configured = True
