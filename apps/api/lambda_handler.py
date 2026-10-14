"""AWS Lambda handler: a thin adapter from the event to backend.entrypoint.ask_question.

Event:
    {"question": "...", "session_id": "...", "user_id": "...", "group_name": "aol.com",
     "document_filenames": ["rates.txt", ...],
     "chat_history": [{"role": "user" | "assistant", "content": "..."}],
     "clarification": "<chosen reading, after a needs_clarification response>",
     "args": {"model": "maverick", "ambiguity": "off" | "assumptions" | "ask", "app_id": "..."}}

Returns the ask_question response (status, status_code, answer or message, sources, limitations, request_id, ...).
Configure storage, trace archive and model settings with the same environment variables as the apps (.env.example).
"""
import logging

from backend.entrypoint import ask_question

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    event = event or {}
    args = event.get('args') or {}
    # Log identifiers only: questions and history can contain PHI.
    logger.info('Request session=%s documents=%s', event.get('session_id'), len(event.get('document_filenames') or []))
    response = ask_question(
        event.get('question'),
        document_filenames=event.get('document_filenames') or [],
        group_name=event.get('group_name'),
        history=event.get('chat_history') or [],
        session_id=event.get('session_id'),
        user_id=event.get('user_id'),
        clarification=event.get('clarification'),
        model=args.get('model'),
        ambiguity=args.get('ambiguity', 'off'),
        app_id=args.get('app_id') or 'lambda',
    )
    logger.info('Response request=%s status=%s', response['request_id'], response['status'])
    return response
