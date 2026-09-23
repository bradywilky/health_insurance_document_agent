"""Adapt local files to the existing read-only S3 get_object contract."""
import io
import json
from pathlib import Path

import pandas as pd


class LocalS3Client:
    def __init__(self, objects, bucket="local"):
        self.objects = objects
        self.bucket = bucket

    def get_object(self, *, Bucket, Key):
        if Bucket != self.bucket or Key not in self.objects:
            raise FileNotFoundError(f"Object not found: s3://{Bucket}/{Key}")
        return {"Body": io.BytesIO(self.objects[Key])}


def prepare_local_file(path, *, encoding="utf-8-sig", delimiter=","):
    path = Path(path).expanduser().resolve(strict=True)
    extension = path.suffix.lower()
    if extension == ".csv":
        frames = {"Sheet1": pd.read_csv(path, encoding=encoding, sep=delimiter)}
    elif extension in {".xlsx", ".xls"}:
        frames = pd.read_excel(path, sheet_name=None)
    else:
        raise ValueError("Expected a .csv, .xlsx, or .xls file")
    base = f"local/preprocessed/{path.name}/"
    objects, sheets = {}, {}
    for name, frame in frames.items():
        frame.columns = frame.columns.map(str)
        key = base + (path.name if extension == ".csv" else name.replace("/", "_") + ".csv")
        if key in objects:
            raise ValueError(f"Sheet names collide after S3 normalization: {name}")
        objects[key] = frame.to_csv(index=False).encode("utf-8")
        # Match types/null counts actually observed by the production CSV loader.
        normalized = pd.read_csv(io.BytesIO(objects[key])) if len(frame.columns) else frame
        sheets[name] = {
            "sheet_type": "data", "row_count": len(normalized),
            "column_count": len(normalized.columns),
            "columns": [{"name": col, "dtype": str(normalized[col].dtype),
                         "null_count": int(normalized[col].isna().sum()),
                         "likely_junk": not col.strip() or col.startswith("Unnamed:")}
                        for col in normalized.columns],
        }
    objects[base + "_metadata.json"] = json.dumps({"sheets": sheets}).encode()
    return dict(s3_client=LocalS3Client(objects), s3_bucket="local", s3_prefix="",
                plan_domain="local", filename=path.name)
