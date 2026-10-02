"""AWS Lambda handler: a thin adapter from the event to backend.entrypoint.ask_question.

Event:
    {"question": "...", "session_id": "...", "document_keys": ["<manifest key>", ...],
     "chat_history": [{"role": "user" | "assistant", "content": "..."}],
     "clarification": "<chosen reading, after a needs_clarification response>",
     "args": {"model": "maverick", "ambiguity": "off" | "assumptions" | "ask", "app_name": "..."}}

Returns the ask_question response (status, status_code, answer or message, sources, limitations, request_id, ...).
Configure storage, audit and model settings with the same environment variables as the apps (.env.example).
"""
import logging

from backend.entrypoint import ask_question

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    event = event or {}
    args = event.get('args') or {}
    # Log identifiers only: questions and history can contain PHI.
    logger.info('Request session=%s documents=%s', event.get('session_id'), len(event.get('document_keys') or []))
    response = ask_question(
        event.get('question'),
        document_keys=event.get('document_keys') or [],
        history=event.get('chat_history') or [],
        session_id=event.get('session_id'),
        clarification=event.get('clarification'),
        model=args.get('model'),
        ambiguity=args.get('ambiguity', 'off'),
        app=args.get('app_name') or 'lambda',
    )
    logger.info('Response request=%s status=%s', response['request_id'], response['status'])
    return response
