import time
import os
from botocore.config import Config
from dataclasses import dataclass, field
from datetime import datetime, timezone

import boto3


@dataclass
class LLMCallLog:
    records: list[dict] = field(default_factory=list)

    def append(self, record: dict) -> None:
        self.records.append(record)


class LLM:

    def __init__(
        self,
        agent_name: str | None = None,
        tool_name: str | None = None,
        model_id: str | None = None,
        params: dict | None = None,
        call_log: LLMCallLog | None = None,
    ):
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.model_id = model_id
        self.params = params
        self.call_log = call_log
        self.output_raw: str | None = None
        self.usage: dict = {}

    def converse(self, system, messages, tool_config=None):
        """Preserve Bedrock toolUse blocks and stop reason; reuse this instance's client."""
        import copy
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
        start = time.monotonic()
        response = self._client.converse(**request)
        self.usage = response.get('usage', {})
        if self.call_log is not None:
            self.call_log.append(dict(timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=round((time.monotonic()-start)*1000), agent_name=self.agent_name,
                tool_name=self.tool_name, model_id=self.model_id, system=system,
                messages=copy.deepcopy(messages), output_raw=copy.deepcopy(response['output']['message']),
                stop_reason=response.get('stopReason'), usage=self.usage))
        return response

    def run(self, system=None, messages=None) -> tuple:
        """Return plain text for synthesis/extraction; tool planning uses converse()."""
        response = self.converse(system, messages)
        if response.get('stopReason') not in {'end_turn', 'stop_sequence'}:
            raise ValueError('Incomplete text response from Bedrock: '+str(response.get('stopReason')))
        self.output_raw = "\n".join(block['text'] for block in response['output']['message']['content']
                                     if 'text' in block)
        if not self.output_raw:
            raise ValueError('Bedrock returned no text')
        return self.output_raw, self.usage
