"""Deterministic local preprocessing into the existing per-sheet CSV contract."""
import argparse
import csv
import io
import json
import re
from pathlib import Path

import pandas as pd

from backend.preprocessing.profiling import column_quality


def _present(value):
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _text(value):
    return str(value).strip() if _present(value) else ""


def _names(values):
    used, names = set(), []
    for index, value in enumerate(values, 1):
        base = _text(value) or f"Unnamed: {index}"
        name, suffix = base, 2
        while name in used:
            name = f"{base}__{suffix}"
            suffix += 1
        used.add(name)
        names.append(name)
    return names


def _infer(series):
    values = series.map(lambda v: v if _present(v) else None)
    populated = values.dropna()
    # Preserve textual identifiers (0012), mixed units, and literal NA codes.
    if len(populated) and all(re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", str(v)) for v in populated):
        return pd.to_numeric(values)
    return values.astype("string")


def _profile(frame):
    return [{"name": str(col), "dtype": str(frame[col].dtype),
             "null_count": int(frame[col].isna().sum()),
             "distinct_count": int(frame[col].nunique()),
             "likely_junk": bool(str(col).startswith("Unnamed:") and frame[col].isna().all()),
             "top_values": [{"value": str(v), "count": int(n)}
                            for v, n in frame[col].value_counts().head(8).items()],
             "value_breakdown_truncated": frame[col].nunique() > 8,
             **column_quality(frame[col])}
            for col in frame.columns]


def normalize_sheet(name, raw, override=None):
    override = override or {}
    raw = raw.copy().where(raw.notna(), None)
    counts = raw.apply(lambda row: sum(_present(v) for v in row), axis=1)
    candidates = [i for i in range(min(30, len(raw))) if counts.iloc[i] >= 2
                  and all(isinstance(v, str) for v in raw.iloc[i] if _present(v))]
    header = next((i for i in candidates if counts.iloc[i] >= max(counts.max() * .6, 2)), None)
    if header is None and len(raw.columns) == 1 and len(raw) > 1 and len(_text(raw.iloc[0, 0])) < 40:
        header = 0
    kind = "metadata" if (re.search(r"about|read.?me|version|history|notes|instructions", name, re.I)
                           or header is None) else "data"
    kind = override.get("sheet_type", kind)
    if kind not in {"data", "metadata"}:
        raise ValueError(f"Invalid sheet_type for {name}")
    if "header_row" in override:
        header = int(override["header_row"]) - 1
        if not 0 <= header < len(raw):
            raise ValueError(f"header_row outside sheet: {name}")
    warnings, context, source_rows = [], [], []
    if kind == "metadata":
        records = [{"source_row": i + 1, "source_column": j + 1, "text": str(value)}
                   for i, row in enumerate(raw.itertuples(index=False, name=None))
                   for j, value in enumerate(row) if _present(value)]
        frame = pd.DataFrame(records, columns=["source_row", "source_column", "text"])
        source_rows = sorted({r["source_row"] for r in records})
        header = None
    else:
        if header is None:
            raise ValueError(f"No header found for {name}; supply header_row override")
        names = _names(raw.iloc[header])
        if len(set(_text(v) for v in raw.iloc[header])) < len(names):
            warnings.append("Blank or duplicate headers were assigned unique names.")
        records = []
        repeated, blank = [], []
        for i, row in enumerate(raw.itertuples(index=False, name=None)):
            values = list(row)
            if i <= header:
                if i < header and any(_present(v) for v in values):
                    context.append({"source_row": i + 1, "values": [_text(v) for v in values]})
                continue
            if not any(_present(v) for v in values):
                blank.append(i + 1)
                continue
            if [_text(v) for v in values] == [_text(v) for v in raw.iloc[header]]:
                repeated.append(i + 1)
                continue
            # Sparse footnotes stay discoverable, but do not count as records.
            if sum(_present(v) for v in values) == 1 and any(
                    _text(v).lower().startswith(("note:", "source:", "footnote:")) for v in values):
                context.append({"source_row": i + 1, "values": [_text(v) for v in values]})
                continue
            records.append(values)
            source_rows.append(i + 1)
        frame = pd.DataFrame(records, columns=names)
        for col in frame:
            frame[col] = _infer(frame[col])
        if repeated:
            warnings.append(f"Repeated header rows excluded: {repeated}")
        if blank:
            warnings.append(f"Blank rows excluded: {blank}")
        if frame.duplicated().any():
            warnings.append(f"Duplicate records retained: {int(frame.duplicated().sum())}")
        warnings.append("Header detection is heuristic; use an override for ambiguous layouts.")
    meta = {"sheet_type": kind, "row_count": len(frame), "column_count": len(frame.columns),
            "columns": _profile(frame), "header_row": header + 1 if header is not None else None,
            "source_row_count": len(raw), "source_rows": source_rows,
            "context_rows": context, "warnings": warnings,
            "classification_method": "override" if override else "heuristic"}
    return frame, meta


def preprocess_file(path, *, encoding="utf-8-sig", delimiter=",", overrides=None):
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix.lower() == ".csv":
        with path.open(encoding=encoding, newline="") as stream:
            rows = list(csv.reader(stream, delimiter=delimiter))
        raw_sheets = {"Sheet1": pd.DataFrame(rows)}
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        raw_sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object, keep_default_na=False)
    else:
        raise ValueError("Expected .csv, .xlsx, or .xls")
    overrides = overrides or {}
    unknown = set(overrides) - set(raw_sheets)
    if unknown:
        raise ValueError(f"Override names not in workbook: {sorted(unknown)}")
    objects, sheets = {}, {}
    for name, raw in raw_sheets.items():
        frame, meta = normalize_sheet(name, raw, overrides.get(name))
        csv_name = path.name if path.suffix.lower() == ".csv" else name.replace("/", "_") + ".csv"
        if Path(csv_name).name != csv_name or "\\" in csv_name or csv_name in objects:
            raise ValueError(f"Unsafe or colliding sheet filename: {name}")
        objects[csv_name] = frame.to_csv(index=False).encode("utf-8")
        sheets[name] = {**meta, "csv_file": csv_name}
    metadata = {"schema_version": 2, "filename": path.name, "sheets": sheets}
    objects["_metadata.json"] = json.dumps(metadata, indent=2, allow_nan=False).encode("utf-8")
    return objects, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overrides", type=Path)
    args = parser.parse_args()
    overrides = json.loads(args.overrides.read_text()) if args.overrides else None
    objects, metadata = preprocess_file(args.file, overrides=overrides)
    # A new directory prevents stale sheets from surviving a subsequent export.
    args.output.mkdir(parents=True, exist_ok=False)
    for name, body in objects.items():
        (args.output / name).write_bytes(body)
    print(json.dumps({name: {k: s[k] for k in ("sheet_type", "row_count", "header_row")}
                      for name, s in metadata["sheets"].items()}, indent=2))


if __name__ == "__main__":
    main()
