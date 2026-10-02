"""Read preprocessed table CSVs through either local or S3 storage."""
import io
import pandas as pd


def load_table(
    s3_client,
    s3_bucket: str,
    s3_prefix: str,
    plan_domain: str,
    filename: str,
    sheet_name: str | None = None,
    sheet_meta: dict | None = None,
) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        key = f"{s3_prefix}{plan_domain}/preprocessed/{filename}/{filename}"
        with s3_client.get_object(Bucket=s3_bucket, Key=key)["Body"] as body:
            raw = body.read()
    else:
        safe_name = (sheet_name or "Sheet1").replace("/", "_")
        csv_key = f"{s3_prefix}{plan_domain}/preprocessed/{filename}/{safe_name}.csv"
        with s3_client.get_object(Bucket=s3_bucket, Key=csv_key)["Body"] as body:
            raw = body.read()
    try:
        text_types = {c["name"]: "string" for c in (sheet_meta or {}).get("columns", [])
                      if c.get("dtype") in {"str", "string", "object"}}
        df = pd.read_csv(io.BytesIO(raw), dtype=text_types or None,
                         keep_default_na=False, na_values=[""])
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()

    return df[[c for c in df.columns if str(c).strip() != ""]]
