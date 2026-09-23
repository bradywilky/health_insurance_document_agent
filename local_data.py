"""Adapt local files to the existing read-only S3 get_object contract."""
import io
import json
from pathlib import Path

class LocalS3Client:
    def __init__(self, objects, bucket="local"):
        self.objects = objects
        self.bucket = bucket

    def get_object(self, *, Bucket, Key):
        if Bucket != self.bucket or Key not in self.objects:
            raise FileNotFoundError(f"Object not found: s3://{Bucket}/{Key}")
        return {"Body": io.BytesIO(self.objects[Key])}


def prepare_local_file(path, *, encoding="utf-8-sig", delimiter=",", overrides=None):
    from preprocess import preprocess_file
    path = Path(path).expanduser().resolve(strict=True)
    objects, _ = preprocess_file(path, encoding=encoding, delimiter=delimiter, overrides=overrides)
    base = f"local/preprocessed/{path.name}/"
    return dict(s3_client=LocalS3Client({base + k: v for k, v in objects.items()}),
                s3_bucket="local", s3_prefix="", plan_domain="local", filename=path.name)


def prepare_preprocessed_directory(path):
    """Read an exported preprocessing directory without reprocessing the workbook."""
    root = Path(path).expanduser().resolve(strict=True)
    meta_bytes = (root / '_metadata.json').read_bytes()
    metadata = json.loads(meta_bytes)
    filename = metadata['filename']
    base = f'local/preprocessed/{filename}/'
    objects = {base + '_metadata.json': meta_bytes}
    for sheet in metadata['sheets'].values():
        name = sheet['csv_file']
        if '/' in name or '\\' in name or Path(name).name != name:
            raise ValueError('CSV file must be a basename inside the preprocessing directory')
        target = (root / name).resolve(strict=True)
        if target.parent != root:
            raise ValueError('CSV path escapes the preprocessing directory')
        objects[base + name] = target.read_bytes()
    return dict(s3_client=LocalS3Client(objects), s3_bucket='local', s3_prefix='',
                plan_domain='local', filename=filename)
