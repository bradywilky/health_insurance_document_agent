"""Convert table results to JSON-compatible values without losing decimals."""
from decimal import Decimal
import pandas as pd

def sanitize_result(obj):
    if isinstance(obj, Decimal):
        return format(obj, 'f')
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, dict):
        return {k: sanitize_result(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_result(v) for v in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    if hasattr(obj, "item"):
        return sanitize_result(obj.item())
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj
