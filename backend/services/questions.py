"""Answer a question with an audit record and a trace. Apps and the CLI call this instead of the agent."""
import time

from backend.agents import document_agent
from backend.agents.llm import LLMCallLog
from backend.observability import audit, tracing


def answer_question(documents, document_ids, question, *, app, session_id=None, sink=None, history=None,
                    call_log=None, model=None, ambiguity='off', clarification=None, on_step=None):
    """Run the document agent. Returns its result plus request_id, trace_id and usage.

    The audit record is written whether the question succeeds or fails; failures are then re-raised."""
    sink = sink if sink is not None else audit.configured_audit_sink()
    log = call_log if call_log is not None else LLMCallLog()
    request_id, timestamp, start = audit.new_id(), audit.now(), time.monotonic()
    docs = [audit.document_summary(documents[d]) for d in document_ids if d in documents]
    result, error = None, None
    with tracing.span('invoke_agent document_agent', **{
            'gen_ai.operation.name': 'invoke_agent', 'gen_ai.agent.name': 'document_agent',
            'session.id': session_id, 'app.request_id': request_id, 'app.name': app,
            'app.ambiguity_mode': ambiguity, 'app.model': model,
            'app.document_ids': [d['id'] for d in docs]}) as root:
        trace_id = tracing.trace_id_of(root)
        tracing.add_content(root, 'app.question', question=question, clarification=clarification)
        try:
            # Module attribute, so tests can replace the agent.
            result = document_agent.run_document_agent(
                documents, document_ids, question, history=history, call_log=log, on_step=on_step, model=model,
                ambiguity=ambiguity, clarification=clarification)
            tracing.set_attributes(root, **{'app.status': result.get('status'),
                                            'app.evidence_count': len(result.get('evidence') or []),
                                            'app.limitations_count': len(result.get('limitations') or [])})
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'[:1000]
            tracing.mark_error(root, error)
            raise
        finally:
            records = audit.question_records(
                request_id=request_id, timestamp=timestamp, app=app, session_id=session_id,
                model=model or 'default', ambiguity=ambiguity, prompt_version=document_agent.PROMPT_VERSION,
                documents=docs, question=question, clarification=clarification, result=result,
                call_records=log.records, duration_ms=round((time.monotonic() - start) * 1000), error=error,
                trace_id=trace_id)
            audit.write(sink, records)
    return {**result, 'request_id': request_id, 'trace_id': trace_id, 'usage': audit.totals(log.records)}


def record_document_event(event, *, app, sink=None, **fields):
    """Audit an upload or profile change. Never raises unless AUDIT_REQUIRED=true."""
    sink = sink if sink is not None else audit.configured_audit_sink()
    from opentelemetry import trace
    audit.write(sink, audit.document_event(event, app=app,
                                           trace_id=tracing.trace_id_of(trace.get_current_span()), **fields))
