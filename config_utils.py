"""Local defaults; deployment may retain its existing config_utils implementation."""
import os

MODELS = {
    "maverick": "us.meta.llama4-maverick-17b-instruct-v1:0",
    "scout": "us.meta.llama4-scout-17b-instruct-v1:0",
}
BUSINESS_CONTEXT = os.getenv("BUSINESS_CONTEXT", "")


def get_model_config(alias="UNALIASED"):
    model = os.getenv("TABLES_MODEL", "maverick").lower()
    if model not in MODELS:
        raise ValueError("TABLES_MODEL must be maverick or scout")
    return os.getenv("BEDROCK_MODEL_ID") or MODELS[model], {
        "maxTokens": int(os.getenv("BEDROCK_MAX_TOKENS", "4096")),
        "temperature": 0.0,
    }
