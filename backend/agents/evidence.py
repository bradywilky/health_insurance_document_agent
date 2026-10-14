"""Compact source-backed evidence and deterministic numeric display checks."""
import json
import re
from decimal import Decimal


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)


SHORT_LIST_CHARS = 2000  # lists of plain values this small (e.g. column names) are never cut


def bounded(value, *, max_chars=18000):
    """Keep valid JSON and explicitly mark any payload removed from model context."""
    if len(compact(value)) <= max_chars:
        return value
    def shrink(obj):
        if isinstance(obj, str):
            return obj[:1200] + (' [text truncated]' if len(obj) > 1200 else '')
        if isinstance(obj, list):
            if all(not isinstance(v, (dict, list)) for v in obj) and len(compact(obj)) <= SHORT_LIST_CHARS:
                return obj
            return [shrink(v) for v in obj[:10]] + ([{'truncated_items': len(obj)-10}] if len(obj)>10 else [])
        if isinstance(obj, dict):
            return {k: shrink(v) for k, v in obj.items()}
        return obj
    preview = shrink(value)
    if len(compact(preview)) > max_chars:
        return {'truncated': True, 'message': 'Result exceeds context budget; query narrower columns/rows.'}
    return {'truncated': True, 'preview': preview}


def bounded_items(items, *, max_chars):
    """Fit a list of evidence items into max_chars by trimming only the largest ones.

    Items smaller than an equal share of what remains are kept whole, so one large result cannot cause a small
    one (such as a sheet list) to be cut."""
    sizes = [len(compact(item)) for item in items]
    if sum(sizes) <= max_chars:
        return items
    remaining, cap = max_chars, None
    for k, size in enumerate(sorted(sizes)):
        share = remaining // (len(sizes) - k)
        if size > share:
            cap = share
            break
        remaining -= size
    return [item if cap is None or size <= cap else bounded(item, max_chars=cap)
            for item, size in zip(items, sizes)]


def numeric_display_check(answer, evidence):
    """Ensure unrounded aggregate values are visible; not a semantic truth verifier."""
    numbers = set()
    for match in re.finditer(r'(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?(?!\w|\.\d)', answer):
        try:
            numbers.add(Decimal(match.group().replace(',', '')))
        except Exception:
            pass
    missing = []
    for record in evidence:
        data = record.get('data', {})
        if not isinstance(data, dict) or not data.get('aggregated') or data.get('error'):
            continue
        for row in data.get('rows', []):
            for col in data.get('decimal_columns', []):
                value = row.get(col)
                if value is not None and Decimal(str(value)) not in numbers:
                    label = ', '.join(f'{k}={v}' for k, v in row.items() if k not in data['decimal_columns'])
                    missing.append(f"- {col}{' (' + label + ')' if label else ''}: {value} [{record['id']}]")
    if missing:
        answer += '\n\nExact computed values:\n' + '\n'.join(dict.fromkeys(missing))
    return answer
