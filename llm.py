import json
import time
import os
from botocore.config import Config
from dataclasses import dataclass, field
from datetime import datetime, timezone

import boto3
import dirtyjson


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
        self.output_transformed: dict | None = None
        self.usage: dict = {}

    def _invoke_llm(self, system: str, messages: list) -> None:
        session = boto3.Session(profile_name=os.getenv("AWS_PROFILE") or None)
        bedrock = session.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION") or session.region_name or "us-east-1",
            config=Config(connect_timeout=10, read_timeout=180,
                          retries={"mode": "standard", "max_attempts": 3}),
        )
        response = bedrock.converse(
            modelId=self.model_id,
            messages=messages,
            system=[{"text": system}],
            inferenceConfig=self.params,
        )
        self.usage = response.get("usage", {})
        self.output_raw = "\n".join(
            block["text"] for block in response["output"]["message"]["content"]
            if "text" in block
        )
        if not self.output_raw:
            raise ValueError("Bedrock returned no text")

    def _parse_llm_json(self, raw: str) -> None:
        try:
            self.output_transformed = dirtyjson.loads(
                raw, search_for_first_object=True
            )
            return
        except Exception:
            pass

        repaired = raw.rstrip()
        for _ in range(3):
            try:
                self.output_transformed = json.loads(repaired)
                return
            except json.JSONDecodeError:
                repaired += "}"

    def run(
        self,
        system: str | None = None,
        messages: list | None = None,
        return_json: bool = False,
    ) -> tuple:
        """Returns (output, usage)."""
        self.output_transformed = None
        t0 = time.monotonic()
        self._invoke_llm(system, messages)
        duration_ms = round((time.monotonic() - t0) * 1000)

        if return_json:
            self._parse_llm_json(self.output_raw)

        if self.call_log is not None:
            self.call_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": duration_ms,
                "agent_name": self.agent_name,
                "tool_name": self.tool_name,
                "model_id": self.model_id,
                "system": system,
                "messages": messages,
                "output_raw": self.output_raw,
                "output_transformed": self.output_transformed,
                "usage": self.usage,
            })

        return (
            self.output_transformed
            if self.output_transformed is not None
            else self.output_raw,
            self.usage,
        )
