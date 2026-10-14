"""Optional reasoning step before each planner call (PLANNER_REASONING=true, or reasoning=True).

The planner model rarely explains its tool calls. When this step is on, the same model is first asked in plain
text, with no tools attached, what the next step should be and why. Its reply is added to the tool conversation
as the planned next step, recorded in the result's `reasoning`, and traced as a `Planner Reasoning` llm span.
With the step off, research runs exactly as before.
"""
import json
import os

RULES = '''
REASONING STEP. You are not calling tools in this reply. Reply in plain text, at most four sentences:
1. What the results so far establish, or "Nothing yet" before the first call.
2. The single next tool call: the tool, its key parameters, and why it is the right next step.
If the evidence already answers the question, say the next step is to call answer, and why.
Do not answer the question and do not write tool requests as JSON.'''

NOTE = ('Your next step should be this (from your own reasoning): {plan}\n'
        'Make that call now as a native tool call. If the results above show the plan is wrong, make the better '
        'call instead.')

RESULT_CHARS = 4000  # per tool result in the transcript; full results stay in the tool conversation


def enabled(setting=None):
    """An explicit setting wins; otherwise PLANNER_REASONING (off unless 'true', '1' or 'yes')."""
    if setting is not None:
        return bool(setting)
    return os.getenv('PLANNER_REASONING', '').strip().lower() in {'true', '1', 'yes'}


def _clip(text):
    return text if len(text) <= RESULT_CHARS else text[:RESULT_CHARS] + ' ...[truncated]'


def _earlier_plan(text):
    return text.startswith(NOTE.split('{')[0])


def transcript(messages):
    """The tool conversation as plain text, so it can be sent without tool definitions.

    Earlier plans are left out: the calls that followed them show what was done, and a model shown its first plan
    tends to repeat it word for word.
    """
    lines = []
    for index, message in enumerate(messages):
        for block in message['content']:
            if 'toolUse' in block:
                call = block['toolUse']
                lines.append(f"PLANNER CALLED {call['name']}: {json.dumps(call.get('input'), default=str)}")
            elif 'toolResult' in block:
                for part in block['toolResult']['content']:
                    if 'text' in part and not _earlier_plan(part['text']):
                        lines.append('RESULT: ' + _clip(part['text']))
            elif 'text' in block and _earlier_plan(block['text']):
                continue
            elif 'text' in block:
                if index == 0:
                    lines.append('REQUEST: ' + block['text'])
                elif message['role'] == 'assistant':
                    lines.append('PLANNER WROTE: ' + _clip(block['text']))
                else:
                    lines.append('NOTE: ' + _clip(block['text']))
    return '\n'.join(lines)


def next_step(llm, system, messages):
    """Ask the planner model, without tools, for its next step and the reason for it."""
    return llm.run(system=system + RULES,
                   messages=[{'role': 'user', 'content': [{'text': transcript(messages)}]}]).strip()
