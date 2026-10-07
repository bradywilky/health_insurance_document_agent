import pandas as pd

from backend.shared.table import parse_csv


def as_table(frame):
    """A DataFrame as the question path sees it: written to CSV as preprocessing does, then read back,
    with non-numeric columns kept as text the way sheet metadata marks them."""
    text = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    return parse_csv(frame.to_csv(index=False).encode(), text_columns=text)
