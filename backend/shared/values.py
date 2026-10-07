"""Value checks shared by preprocessing profiles and query diagnostics: possible spelling variants,
missing markers and placeholder numbers. Standard library only, so the question path needs no pandas.
"""
import re
from collections import Counter

# "NA" is excluded on purpose: it is often a real code (e.g. "not applicable").
MISSING_MARKERS = {'nan', 'null', 'none', 'nil', 'n/a', '#n/a', 'missing', '(blank)'}


def normalize_key(value):
    """Case-, space- and punctuation-insensitive comparison key."""
    return re.sub(r'[^0-9a-z]+', '', str(value).casefold())


def _subsequence(short, long):
    rest = iter(long)
    return all(ch in rest for ch in short)


def _similar(raw_a, raw_b):
    """True when two values plausibly name the same thing.

    Deliberately conservative: only case/space/punctuation differences, or dropped letters
    (Vsa/Visa, tehr@n/Tehran, fail/failed). Substitutions are never grouped, because pairs
    such as Medicare/Medicaid, Male/Female and Dental/Rental are different things.
    """
    a, b = normalize_key(raw_a), normalize_key(raw_b)
    if a == b:
        return True
    (short, raw_short), (long, raw_long) = sorted([(a, raw_a), (b, raw_b)], key=lambda p: len(p[0]))
    if len(short) < 3 or short.isdigit() or len(long) - len(short) > 3 or short[0] != long[0]:
        return False
    # A separate qualifier word (Medicaid / Medicaid A, Premium / Premium Plus) is a different value.
    if str(raw_long).strip().casefold().startswith(str(raw_short).strip().casefold() + ' '):
        return False
    if re.sub(r'\d', '', a) == re.sub(r'\d', '', b):  # Tier 1 / Tier 10
        return False
    fewer, more = sorted([_words(raw_a), _words(raw_b)], key=len)
    # An added whole word makes a different element (Account - Date / Account - End Date).
    if len(fewer) < len(more) and _subsequence(fewer, more):
        return False
    # In multi-word names, a word extended at either end is a different term (Pack / Package Amount).
    # Mid-word typos (Acount / Account) and single words (fail / failed) are still grouped.
    if len(fewer) == len(more) >= 2:
        changed = [(x, y) for x, y in zip(fewer, more) if x != y]
        if len(changed) == 1:
            x, y = sorted(changed[0], key=len)
            if y.startswith(x) or y.endswith(x):
                return False
    return _subsequence(short, long)


def _words(raw):
    return re.findall(r'[0-9a-z]+', str(raw).casefold())


def _abbreviation(raw, key, other):
    """All-caps codes such as THR whose letters appear in order in a longer value's key."""
    if not (2 <= len(key) <= 4 and str(raw).strip().isupper() and len(other) > len(key) + 1):
        return False
    if key[0] != other[0]:
        return False
    return _subsequence(key, other)


def variant_groups(counts):
    """Group values (value -> count) that look like spellings of one thing.

    Returns [{'suggested': most frequent value, 'variants': {value: count, ...}}], largest first.
    """
    values = [v for v in counts if str(v).strip().casefold() not in MISSING_MARKERS and normalize_key(v)]
    parent = {v: v for v in values}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    keys = {v: normalize_key(v) for v in values}
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            ka, kb = keys[a], keys[b]
            if _similar(a, b) or _abbreviation(a, ka, kb) or _abbreviation(b, kb, ka):
                parent[find(a)] = find(b)
    groups = {}
    for v in values:
        groups.setdefault(find(v), []).append(v)
    result = []
    for members in groups.values():
        if len(members) > 1:
            # Most frequent first; on ties prefer a trimmed, non-all-caps spelling over a code such as THR.
            members.sort(key=lambda v: (-counts[v], str(v).isupper(), str(v) != str(v).strip()))
            result.append({'suggested': members[0], 'variants': {m: int(counts[m]) for m in members}})
    return sorted(result, key=lambda g: -sum(g['variants'].values()))


def placeholder_values(values):
    """Repeated numbers made only of 9s (e.g. 999, -999999), a common 'unknown' sentinel. Skips missing values."""
    found = {}
    for value, count in Counter(v for v in values if v is not None and v == v).items():
        try:
            digits = format(abs(float(value)), 'f').rstrip('0').rstrip('.')
        except (TypeError, ValueError):
            continue
        if count > 1 and len(digits) >= 3 and set(digits) == {'9'}:
            found[str(value)] = int(count)
    return found


def similar_values(requested, counts):
    """Values in counts (value -> count) that look like spellings of a requested value but were not requested."""
    wanted = set(requested)
    found = {}
    for value, count in counts.items():
        if value in wanted or value.strip().casefold() in MISSING_MARKERS:
            continue
        for target in wanted:
            kv, kt = normalize_key(value), normalize_key(target)
            if kv and kt and (_similar(value, target) or _abbreviation(value, kv, kt)
                              or _abbreviation(target, kt, kv)):
                found[value] = count
                break
    return found
