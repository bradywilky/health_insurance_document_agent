"""Column data-quality hints: possible spelling variants, missing markers and placeholder numbers.

Everything here is a suggestion for the agent to verify and disclose. Nothing is merged or
converted; the exported CSVs keep every original value.
"""
import pandas as pd

from backend.shared.values import MISSING_MARKERS, placeholder_values, variant_groups

MAX_VARIANT_DISTINCT = 200  # skip free-text / identifier-like columns
ALL_VALUES_DISTINCT = 30    # list every value for low-cardinality columns
CATEGORICAL_DISTINCT = 50   # variant checks run on columns with few distinct values (or <= 20% distinct)


def column_quality(series):
    """Extra profile fields for one column; empty dict when nothing is notable."""
    notes = {}
    present = series.dropna()
    if pd.api.types.is_numeric_dtype(series):
        if len(present):
            notes.update(min=str(present.min()), max=str(present.max()),
                         negative_count=int((present < 0).sum()))
        placeholders = placeholder_values(present)
        if placeholders:
            notes['possible_placeholder_values'] = placeholders
        return notes
    counts = present.value_counts()
    markers = {str(v): int(n) for v, n in counts.items() if str(v).strip().casefold() in MISSING_MARKERS}
    if markers:
        notes['possible_missing_markers'] = markers
    # Only category-like columns: name lists and free-text definitions are mostly unique, and their
    # near-duplicates are usually distinct entries rather than misspellings.
    categorical = len(counts) <= CATEGORICAL_DISTINCT or len(counts) <= 0.2 * len(present)
    short_values = counts.index.map(lambda v: len(str(v))).to_series().mean() <= 40 if len(counts) else False
    if 1 < len(counts) <= MAX_VARIANT_DISTINCT and categorical and short_values:
        groups = variant_groups({str(v): int(n) for v, n in counts.items()})
        if groups:
            notes['possible_variant_groups'] = groups
    if len(counts) <= ALL_VALUES_DISTINCT:
        notes['all_values'] = {str(v): int(n) for v, n in counts.items()}
    else:
        # Status words inside a column too varied to list in full (e.g. "Out of Scope" in a list of field names).
        threshold = max(3, 0.02 * len(present))
        frequent = {str(v): int(n) for v, n in counts.head(8).items() if n >= threshold}
        if frequent:
            notes['frequent_values'] = frequent
    return notes
