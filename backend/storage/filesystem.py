"""Persistent local object storage shared by the upload and chat processes."""
from pathlib import Path
import os
import shutil
import tempfile
from backend.storage.documents import ObjectDocumentStore, PREPROCESSED


class DirectoryClient:
    def __init__(self, root, *, read_only=False):
        self.root = Path(root).resolve()
        self.read_only = read_only

    def _path(self, key):
        if not key or "\\" in key or ":" in key or key.startswith("/") or any(
            part in {".", "..", ""} for part in key.split("/")
        ):
            raise ValueError("Invalid local object key")
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Object key escapes the document library")
        return path

    def get_object(self, *, Bucket, Key):
        return {"Body": self._path(Key).open("rb")}

    def upload_fileobj(self, stream, bucket, key, ExtraArgs=None):
        if self.read_only:
            raise PermissionError("This document store is read-only")
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, prefix=".upload-") as out:
                temporary = out.name
                shutil.copyfileobj(stream, out)
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def delete_object(self, *, Bucket, Key):
        if self.read_only:
            raise PermissionError("This document store is read-only")
        self._path(Key).unlink(missing_ok=True)

    def get_paginator(self, operation):
        if operation != "list_objects_v2":
            raise ValueError("Unsupported operation")
        return self

    def paginate(self, *, Bucket, Prefix):
        folder = self._path(Prefix.rstrip("/"))
        objects = []
        if folder.exists():
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    key = path.relative_to(self.root).as_posix()
                    self._path(key)
                    objects.append({"Key": key})
        yield {"Contents": objects}


class LocalDocumentStore(ObjectDocumentStore):
    def __init__(self, directory="data/document_library", *, group=None, preprocessed=PREPROCESSED, read_only=False):
        root = Path(directory).expanduser()
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[2] / root
        self.directory = root.resolve()
        super().__init__(DirectoryClient(self.directory, read_only=read_only),
                         "local", prefix="", group=group, preprocessed=preprocessed, read_only=read_only)
