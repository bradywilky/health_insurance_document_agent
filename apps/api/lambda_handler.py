"""AWS Lambda handler: a thin adapter from the event to backend.entrypoint.ask_question.

Event:
    {"question": "...", "session_id": "...", "document_keys": ["<saved filename>", ...],
     "chat_history": [{"role": "user" | "assistant", "content": "..."}],
     "clarification": "<chosen reading, after a needs_clarification response>",
     "args": {"model": "maverick", "ambiguity": "off" | "assumptions" | "ask", "app_name": "..."}}

Returns the ask_question response (status, status_code, answer or message, sources, limitations, request_id, ...).
Configure storage, audit and model settings with the same environment variables as the apps (.env.example).
"""
import functools
import logging

from backend.entrypoint import ask_question
from backend.observability.audit import configured_audit_sink
from backend.storage.s3 import configured_store

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@functools.lru_cache(maxsize=1)
def clients():
    """Created on the first request and reused while the Lambda stays warm."""
    return configured_store(read_only=True), configured_audit_sink()


def lambda_handler(event, context):
    event = event or {}
    args = event.get('args') or {}
    # Log identifiers only: questions and history can contain PHI.
    logger.info('Request session=%s documents=%s', event.get('session_id'), len(event.get('document_keys') or []))
    store, sink = clients()
    response = ask_question(
        event.get('question'),
        document_keys=event.get('document_keys') or [],
        history=event.get('chat_history') or [],
        session_id=event.get('session_id'),
        clarification=event.get('clarification'),
        model=args.get('model'),
        ambiguity=args.get('ambiguity', 'off'),
        app=args.get('app_name') or 'lambda',
        store=store,
        sink=sink,
    )
    logger.info('Response request=%s status=%s', response['request_id'], response['status'])
    return response
