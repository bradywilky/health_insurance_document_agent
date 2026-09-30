"""Shared native Bedrock protocol transport and action validation."""
import json
from jsonschema import Draft202012Validator

def validate_action(action, specs):
    if not isinstance(action.get('tool'), str):
        raise ValueError('Tool name must be a string')
    spec = next((s['toolSpec'] for s in specs if s['toolSpec']['name']==action['tool']), None)
    if spec is None:
        raise ValueError('Unknown tool name')
    errors = list(Draft202012Validator(spec['inputSchema']['json']).iter_errors(action.get('parameters')))
    if errors:
        raise ValueError(errors[0].message[:800])


class NativePlanner:
    def __init__(self, llm, system, payload, specs):
        self.llm, self.system = llm, system
        self.config = {'tools':specs}
        self.messages = [{'role':'user','content':[{'text':json.dumps(payload,default=str)}]}]
        self.pending = []

    def request(self):
        response = self.llm.converse(self.system, self.messages, self.config)
        message = response['output']['message']
        self.pending = [b['toolUse'] for b in message['content'] if 'toolUse' in b]
        ids = [t.get('toolUseId') for t in self.pending]
        if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):
            self.pending = []
            raise ValueError('Invalid or duplicate native toolUse IDs')
        # Preserve complete assistant blocks and pair every toolUse with one result.
        self.messages.append(message)
        if response.get('stopReason')!='tool_use' or not self.pending:
            raise ValueError('Research must return native tool calls; prose is not a tool result. Stop reason: '+str(response.get('stopReason')))
        return [{'tool':t['name'],'parameters':t.get('input')} for t in self.pending]

    def feedback(self, results, error=None):
        if self.pending:
            content = []
            for i,tool in enumerate(self.pending):
                result = results[i] if i<len(results) else {'error':error or 'Request was not executed'}
                # Text is portable across the Llama and Claude Converse adapters.
                content.append({'toolResult':{'toolUseId':tool['toolUseId'],
                    'content':[{'text':json.dumps(result,default=str)}]}})
            self.messages.append({'role':'user','content':content})
        else:
            self.messages.append({'role':'user','content':[{'text':error or 'Return a native tool call.'}]})
        self.pending = []

    def note(self, text):
        """Add guidance after the latest tool results, keeping user/assistant turns alternating.

        Bedrock rejects text blocks beside toolResult blocks, so the note joins the last result's content.
        """
        last = self.messages[-1]
        if last['role'] == 'user' and 'toolResult' in last['content'][-1]:
            last['content'][-1]['toolResult']['content'].append({'text': text})
        elif last['role'] == 'user':
            last['content'].append({'text': text})
        else:
            self.messages.append({'role': 'user', 'content': [{'text': text}]})
