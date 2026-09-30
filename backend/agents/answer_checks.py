"""Deterministic checks on a written answer. They flag likely errors; they do not prove correctness."""
import re
from decimal import Decimal, InvalidOperation

from backend.agents.evidence import compact
from backend.tools.calculations import extract_numbers

CURRENCY = re.compile(r'(?:USD|US\$|\$)\s?(-?\d[\d,]*(?:\.\d+)?)|(-?\d[\d,]*(?:\.\d+)?)\s?(?:USD|dollars)\b')
CERTAINTY = re.compile(
    r'\b(?:(?:insurer|plan|carrier|payer|bingle-dingle(?: insurance)?|we)\s+(?:will|shall|is going to)\s+(?:pay|reimburse)'
    r'|will be (?:paid|reimbursed)|final (?:payment|reimbursement) (?:is|will be|of|amount is))', re.I)
HEDGE = re.compile(r"\b(?:cannot|can't|can ?not|not|unable|unknown|undetermined|whether|how much|what)\b|\?", re.I)


def unsupported_amounts(answer, evidence, question):
    """Currency amounts in the answer that match no number in the question, evidence or calculations."""
    supported = extract_numbers(question) | extract_numbers(compact(evidence))
    missing = []
    for match in CURRENCY.finditer(answer):
        raw = match.group(1) or match.group(2)
        try:
            value = Decimal(raw.replace(',', ''))
        except InvalidOperation:
            continue
        if value not in supported and raw not in missing:
            missing.append(raw)
    return missing


def payment_certainty(answer):
    """Sentences asserting what will be paid; this application never holds a final payment decision."""
    flagged = []
    for sentence in re.split(r'(?<=[.!?])\s+|\n+', answer):
        if CERTAINTY.search(sentence) and not HEDGE.search(sentence):
            flagged.append(sentence.strip()[:300])
    return flagged


def answer_issues(answer, evidence, question):
    issues = []
    amounts = unsupported_amounts(answer, evidence, question)
    if amounts:
        issues.append('These amounts appear in no evidence or calculation result: ' + ', '.join(amounts)
                      + '. Use only evidence values and calculate results; do not restate them differently.')
    for sentence in payment_certainty(answer):
        issues.append(f'States a payment as certain: "{sentence}". Describe scheduled/allowed amounts and '
                      'estimates as such; the selected documents contain no final payment decision.')
    return issues
