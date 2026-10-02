"""Bedrock model choices and inference settings, read from the environment."""
import os

MODELS = {
    "maverick": "us.meta.llama4-maverick-17b-instruct-v1:0",
    "claude-haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-opus-4.5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
}


def get_model_config():
    model = os.getenv("TABLES_MODEL", "maverick").lower()
    if model not in MODELS:
        raise ValueError(f"TABLES_MODEL must be one of {list(MODELS)}")
    return os.getenv("BEDROCK_MODEL_ID") or MODELS[model], {
        "maxTokens": int(os.getenv("BEDROCK_MAX_TOKENS", "4096")),
        "temperature": 0.0,
    }
