import os
from botocore.config import Config
from dataclasses import dataclass

import boto3

from backend.observability import llmobs


@dataclass
class Usage:
    """Model calls and tokens for one question, returned in the response and the audit record.

    Per-call detail (prompts, responses, timings) goes to Datadog LLM Observability instead."""
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage: dict) -> None:
        self.llm_calls += 1
        self.input_tokens += usage.get('inputTokens', 0)
        self.output_tokens += usage.get('outputTokens', 0)

    def totals(self) -> dict:
        return {'llm_calls': self.llm_calls, 'input_tokens': self.input_tokens, 'output_tokens': self.output_tokens}


class LLM:

    def __init__(
        self,
        agent_name: str | None = None,
        tool_name: str | None = None,
        model_id: str | None = None,
        params: dict | None = None,
        usage_meter: Usage | None = None,
    ):
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.model_id = model_id
        self.params = params
        self.usage_meter = usage_meter

    def converse(self, system, messages, tool_config=None):
        """Preserve Bedrock toolUse blocks and stop reason; reuse this instance's client."""
        if not hasattr(self, '_client'):
            session = boto3.Session(profile_name=os.getenv('AWS_PROFILE') or None)
            self._client = session.client('bedrock-runtime',
                region_name=os.getenv('AWS_REGION') or session.region_name or 'us-east-1',
                config=Config(connect_timeout=10, read_timeout=180,
                              # Adaptive mode slows down under throttling; shared accounts hit rate limits.
                              retries={'mode':'adaptive',
                                       'total_max_attempts':int(os.getenv('BEDROCK_MAX_ATTEMPTS', '6'))}))
        request = dict(modelId=self.model_id, system=[{'text':system}],
                       messages=messages, inferenceConfig=self.params)
        if tool_config is not None:
            request['toolConfig'] = tool_config
        # Prompt and response text only when TRACE_CONTENT=true.
        with llmobs.span('llm', self.tool_name or 'chat', model_name=self.model_id,
                         model_provider=llmobs.MODEL_PROVIDER) as current:
            response = self._client.converse(**request)
            usage = response.get('usage', {})
            message = response['output']['message']
            llmobs.annotate(current, metadata={
                'agent': self.agent_name, 'stop_reason': response.get('stopReason'),
                'max_tokens': (self.params or {}).get('maxTokens'),
                'tools_offered': len((tool_config or {}).get('tools', [])),
                'tool_calls': [b['toolUse']['name'] for b in message.get('content', []) if 'toolUse' in b]},
                metrics={'input_tokens': usage.get('inputTokens'), 'output_tokens': usage.get('outputTokens'),
                         'total_tokens': usage.get('inputTokens', 0) + usage.get('outputTokens', 0)})
            llmobs.annotate_content(current, input_data=llmobs.bedrock_messages(messages, system),
                                    output_data=llmobs.bedrock_messages([message]))
        if self.usage_meter is not None:
            self.usage_meter.add(usage)
        return response

    def run(self, system=None, messages=None) -> str:
        """Return plain text for synthesis/extraction; tool planning uses converse()."""
        response = self.converse(system, messages)
        if response.get('stopReason') not in {'end_turn', 'stop_sequence'}:
            raise ValueError('Incomplete text response from Bedrock: '+str(response.get('stopReason')))
        text = "\n".join(block['text'] for block in response['output']['message']['content'] if 'text' in block)
        if not text:
            raise ValueError('Bedrock returned no text')
        return text
