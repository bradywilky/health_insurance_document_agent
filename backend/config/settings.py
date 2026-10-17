"""Bedrock model choices and inference settings, read from the environment."""
import os

MODELS = {
    "maverick": "us.meta.llama4-maverick-17b-instruct-v1:0",
    "claude-haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-opus-4.5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "nova-2-lite": "us.amazon.nova-2-lite-v1:0",
}


def get_model_config():
    model = os.getenv("TABLES_MODEL", "maverick").lower()
    if model not in MODELS:
        raise ValueError(f"TABLES_MODEL must be one of {list(MODELS)}")
    return os.getenv("BEDROCK_MODEL_ID") or MODELS[model], {
        "maxTokens": int(os.getenv("BEDROCK_MAX_TOKENS", "4096")),
        "temperature": 0.0,
    }


def semantic_search_enabled():
    """SEMANTIC_SEARCH=true turns on passage embeddings at upload and the semantic_search tool (default off).

    Needs a Bedrock embedding model (EMBEDDING_MODEL_ID); see backend/retrieval/semantic.py for where vectors live.
    """
    return os.getenv('SEMANTIC_SEARCH', 'false').strip().lower() in {'1', 'true', 'yes', 'on'}
