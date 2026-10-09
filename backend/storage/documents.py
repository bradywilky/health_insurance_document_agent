"""Shared object-store layout for originals, extracted blocks, and table CSVs.

Saved files live under <prefix><group>/<preprocessed>/<filename>/ (manifest.json, extracted.json, table CSVs),
and originals under the sibling folder <prefix><group>/<parent of preprocessed>/raw/<filename>. In S3:

    content/health_insurance_document_agent/<group_name>/documents/preprocessed/<filename>/manifest.json
"""
import copy
from io import BytesIO
import json
import mimetypes
import re

PREPROCESSED = 'documents/preprocessed'


def _segments(value, what):
    value = value.strip('/')
    parts = value.split('/') if value else []
    if any(part in {'', '.', '..'} or '\\' in part for part in parts):
        raise ValueError(f'{what} cannot contain empty or dot path segments')
    return parts


def check_group_name(group):
    """A group name is one folder name, e.g. 'aol.com'."""
    if not isinstance(group, str) or not group or group != group.strip() or '/' in group or '\\' in group \
            or group in {'.', '..'}:
        raise ValueError('group_name must be a single folder name, e.g. "aol.com"')
    return group


class ObjectDocumentStore:
    # S3 libraries are always split by group, so ask_question requires a group name for them.
    requires_group = False

    def __init__(self, client, bucket, prefix='', *, group=None, preprocessed=PREPROCESSED, kms_key_id=None,
                 read_only=False):
        if not bucket or '/' in bucket or bucket != bucket.strip():
            raise ValueError('S3 bucket must be a bucket name, not an s3:// URI')
        base = _segments(prefix, 'S3 prefix')
        folders = _segments(preprocessed, 'Preprocessed prefix')
        if len(folders) < 2 or folders[-1] != 'preprocessed':
            raise ValueError("Preprocessed prefix must end in '/preprocessed', e.g. 'documents/preprocessed'")
        self.client, self.bucket = client, bucket
        self.base_prefix = '/'.join(base) + '/' if base else ''
        # The table tools read <s3_prefix><plan_domain>/preprocessed/<filename>/, so keep the parent separately.
        self.domain = '/'.join(folders[:-1])
        self.kms_key_id = kms_key_id
        self.read_only = read_only
        self.group = None if group is None else check_group_name(group)
        self.prefix = self.base_prefix + (self.group + '/' if self.group else '')

    def for_group(self, group):
        """The same store and client, scoped to another group's folder."""
        scoped = copy.copy(self)
        scoped.group = check_group_name(group)
        scoped.prefix = self.base_prefix + scoped.group + '/'
        return scoped

    @property
    def preprocessed(self):
        return f'{self.prefix}{self.domain}/preprocessed/'

    def _write(self, key, body, content_type):
        extra = {'ContentType': content_type}
        # Without an override, preserve the destination bucket's encryption policy.
        if self.kms_key_id:
            extra.update(ServerSideEncryption='aws:kms', SSEKMSKeyId=self.kms_key_id)
        self.client.upload_fileobj(BytesIO(body), self.bucket, key, ExtraArgs=extra)

    def _json(self, key, value):
        self._write(key, json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'),
                    'application/json')

    def _read_json(self, key):
        with self.client.get_object(Bucket=self.bucket, Key=key)['Body'] as body:
            raw = body.read(50 * 1024 * 1024 + 1)
        if len(raw) > 50 * 1024 * 1024:
            raise ValueError('Saved metadata exceeds the supported size')
        return json.loads(raw)

    def _root(self, manifest_key):
        pattern = re.escape(self.preprocessed) + r'[^/\\]+/manifest\.json'
        if not isinstance(manifest_key, str) or not re.fullmatch(pattern, manifest_key):
            raise ValueError('Manifest must be inside the configured document prefix')
        root = manifest_key[:-len('manifest.json')]
        if root.split('/')[-2] in {'.', '..'}:
            raise ValueError('Invalid saved filename')
        return root

    def save(self, doc, content):
        if self.read_only:
            raise PermissionError("This document store is read-only")
        from hashlib import sha256
        if doc.id != sha256(doc.name.encode() + b'\0' + content).hexdigest()[:16]:
            raise ValueError('Original bytes do not match the processed document')
        if not doc.name or '/' in doc.name or '\\' in doc.name or doc.name in {'.', '..'}:
            raise ValueError('Document filename must be a basename')
        root = f'{self.preprocessed}{doc.name}/'
        # A filename identifies one current document. Invalidate completion before replacing it.
        self.client.delete_object(Bucket=self.bucket, Key=root + 'manifest.json')
        pages = self.client.get_paginator('list_objects_v2').paginate(Bucket=self.bucket, Prefix=root)
        stale = [obj['Key'] for page in pages for obj in page.get('Contents', [])]
        for key in stale:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        self._write(f'{self.prefix}{self.domain}/raw/' + doc.name, content,
                    mimetypes.guess_type(doc.name)[0] or 'application/octet-stream')
        if doc.table_inputs:
            inputs = doc.table_inputs
            source_base = f"{inputs['s3_prefix']}{inputs['plan_domain']}/preprocessed/{doc.name}/"
            names = ['_metadata.json'] + [s['csv_file'] for s in doc.table_metadata['sheets'].values()]
            for name in dict.fromkeys(names):
                if '/' in name or '\\' in name or name in {'.', '..'}:
                    raise ValueError('Preprocessed object name must be a basename')
                with inputs['s3_client'].get_object(Bucket=inputs['s3_bucket'], Key=source_base+name)['Body'] as stream:
                    data = stream.read()
                self._write(root + name, data,
                            'application/json' if name.endswith('.json') else 'text/csv')
        self._json(root + 'extracted.json', {'blocks':doc.blocks, 'warnings':doc.warnings,
                                            'table_metadata':doc.table_metadata})
        if doc.profile:
            self._json(root + 'semantic_profile.json', doc.profile)
        # Publish the manifest last. Failed uploads do not appear in the saved-file picker.
        key = root + 'manifest.json'
        manifest = {'schema_version':2, 'document_id':doc.id, 'filename':doc.name,
                    'kind':doc.kind, 'has_tables':bool(doc.table_inputs), 'has_profile':bool(doc.profile)}
        self._json(key, manifest)
        doc.storage_ref = {'bucket':self.bucket, 'manifest_key':key}
        return doc.storage_ref

    def manifest_key(self, filename):
        """Manifest key for a saved filename. A full manifest key is returned unchanged."""
        if isinstance(filename, str) and filename.endswith('/manifest.json'):
            return filename
        if not isinstance(filename, str) or not filename or '/' in filename or '\\' in filename \
                or filename in {'.', '..'}:
            raise ValueError('Document must be a saved filename or manifest key')
        return f'{self.preprocessed}{filename}/manifest.json'

    def load(self, filename):
        """Load a saved document by filename (e.g. 'rates.txt') or by its full manifest key."""
        from backend.shared.models import Document
        manifest_key = self.manifest_key(filename)
        root = self._root(manifest_key)
        manifest = self._read_json(manifest_key)
        if manifest.get('schema_version') != 2:
            raise ValueError('Unsupported saved document version')
        name = manifest['filename']
        if not name or '/' in name or '\\' in name or name in {'.', '..'}:
            raise ValueError('Invalid saved filename')
        if name != root.split('/')[-2]:
            raise ValueError('Filename does not match its storage path')
        extracted = self._read_json(root + 'extracted.json')
        doc = Document(manifest['document_id'], name, manifest['kind'],
                       blocks=extracted['blocks'], warnings=extracted['warnings'],
                       table_metadata=extracted.get('table_metadata'))
        if manifest['has_tables']:
            if not doc.table_metadata:
                raise ValueError('Saved table metadata is missing')
            for sheet in doc.table_metadata['sheets'].values():
                csv_name = sheet['csv_file']
                if '/' in csv_name or '\\' in csv_name or csv_name in {'.', '..'}:
                    raise ValueError('Invalid saved CSV filename')
            doc.table_inputs = dict(s3_client=self.client, s3_bucket=self.bucket,
                                    s3_prefix=self.prefix, plan_domain=self.domain, filename=name)
        if manifest.get('has_profile'):
            doc.profile = self._read_json(root + 'semantic_profile.json')
        doc.storage_ref = {'bucket':self.bucket, 'manifest_key':manifest_key}
        return doc

    def save_profile(self, manifest_key, profile):
        """Store a generated or reviewed profile for an already saved document."""
        if self.read_only:
            raise PermissionError("This document store is read-only")
        root = self._root(manifest_key)
        manifest = self._read_json(manifest_key)
        if not isinstance(profile, dict) or not profile:
            raise ValueError('Profile must be a nonempty object')
        self._json(root + 'semantic_profile.json', profile)
        self._json(manifest_key, {**manifest, 'has_profile': True})

    def list_documents(self, limit=100):
        if type(limit) is not int or limit < 1:
            raise ValueError('limit must be a positive integer')
        items = []
        pages = self.client.get_paginator('list_objects_v2').paginate(
            Bucket=self.bucket, Prefix=self.preprocessed)
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if not key.endswith('/manifest.json'):
                    continue
                self._root(key)
                item = self._read_json(key)
                if item.get('schema_version') != 2:
                    continue
                items.append({**item, 'manifest_key':key})
                if len(items) == limit:
                    return items
        return items
