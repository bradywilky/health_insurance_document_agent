"""Saved documents, separated by group: save, list, load and delete.

Each group (one organization's users) has its own folder; everyone in a group sees all of its documents.

    <prefix><group>/documents/raw/<filename>                                  the original upload
    <prefix><group>/documents/preprocessed/<filename>/manifest.json           written last; marks it complete
    <prefix><group>/documents/preprocessed/<filename>/extracted.json          text blocks, warnings, sheet metadata
    <prefix><group>/documents/preprocessed/<filename>/<sheet>.csv             one per sheet, for spreadsheets
    <prefix><group>/documents/preprocessed/<filename>/semantic_profile.json   when a profile was generated
    <prefix><group>/documents/preprocessed/<filename>/search_hints.json       when search hints were generated
    <prefix><group>/documents/preprocessed/<filename>/embeddings.json         when passages were embedded: the
                                                                              vectors, or their S3 Vectors keys

The same code serves S3 and the local document library (backend/storage/filesystem.py).
"""
from contextlib import contextmanager
import copy
from io import BytesIO
import json
import mimetypes
import re

from backend.storage.tables import TableFolder

PREPROCESSED = 'documents/preprocessed'


def _segments(value, what):
    value = value.strip('/')
    parts = value.split('/') if value else []
    if any(part in {'', '.', '..'} or '\\' in part for part in parts):
        raise ValueError(f'{what} cannot contain empty or dot path segments')
    return parts


def _basename(name):
    if not isinstance(name, str) or not name or '/' in name or '\\' in name or name in {'.', '..'}:
        raise ValueError('Saved file name must be a basename')
    return name


@contextmanager
def _trace(name, **metadata):
    """A document change is its own archived trace (backend/observability/tracing.py)."""
    from backend.observability import tracing
    from backend.observability.archive import configured_archive
    with tracing.span('workflow', name, archive=configured_archive()) as current:
        tracing.annotate(current, metadata=metadata)
        yield current


def _profile_change(before, after):
    """Glossary terms added, removed, changed and confirmed by a profile save; terms only, no meanings."""
    old = {g['term']: (g.get('meaning'), g.get('source')) for g in before.get('glossary', [])}
    new = {g['term']: (g.get('meaning'), g.get('source')) for g in after.get('glossary', [])}
    meta = after.get('meta', {})
    return {'reviewed': bool(meta.get('reviewed')), 'profile_model_id': meta.get('model_id'),
            'glossary_terms': len(new), 'terms_added': sorted(set(new) - set(old)),
            'terms_removed': sorted(set(old) - set(new)),
            'terms_changed': sorted(t for t in set(new) & set(old) if new[t] != old[t]),
            'terms_confirmed': sum(1 for v in new.values() if v[1] == 'confirmed')}


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
                 read_only=False, vectors=None):
        if not bucket or '/' in bucket or bucket != bucket.strip():
            raise ValueError('S3 bucket must be a bucket name, not an s3:// URI')
        base = _segments(prefix, 'S3 prefix')
        folders = _segments(preprocessed, 'Preprocessed prefix')
        if len(folders) < 2 or folders[-1] != 'preprocessed':
            raise ValueError("Preprocessed prefix must end in '/preprocessed', e.g. 'documents/preprocessed'")
        self.client, self.bucket = client, bucket
        self.base_prefix = '/'.join(base) + '/' if base else ''
        # Originals go in the raw folder beside preprocessed.
        self.library = '/'.join(folders[:-1])
        self.kms_key_id = kms_key_id
        self.read_only = read_only
        # backend/retrieval/semantic.py S3VectorIndex, or None to keep vectors in embeddings.json.
        self.vectors = vectors
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
        return f'{self.prefix}{self.library}/preprocessed/'

    def _raw_key(self, filename):
        return f'{self.prefix}{self.library}/raw/{filename}'

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
        self._remove_outputs(root)
        embeddings = None
        if doc.embeddings:
            try:
                embeddings = self._stored_embeddings(doc.id, doc.name, doc.embeddings)
            except Exception as exc:
                # Like a failed hint batch, a vector index failure never blocks an upload.
                doc.warnings.append(f'Embeddings not saved: {type(exc).__name__}: {exc}')
        self._write(self._raw_key(doc.name), content, mimetypes.guess_type(doc.name)[0] or 'application/octet-stream')
        if doc.table_folder:
            for sheet in doc.table_metadata['sheets'].values():
                name = _basename(sheet['csv_file'])
                self._write(root + name, doc.table_folder.read(name), 'text/csv')
        self._json(root + 'extracted.json', {'blocks':doc.blocks, 'warnings':doc.warnings,
                                            'table_metadata':doc.table_metadata})
        if doc.profile:
            self._json(root + 'semantic_profile.json', doc.profile)
        if doc.search_hints:
            self._json(root + 'search_hints.json', doc.search_hints)
        if embeddings:
            self._json(root + 'embeddings.json', embeddings)
        # Publish the manifest last. Failed uploads do not appear in the saved-file picker.
        key = root + 'manifest.json'
        manifest = {'schema_version':2, 'document_id':doc.id, 'filename':doc.name,
                    'kind':doc.kind, 'has_tables':bool(doc.table_folder), 'has_profile':bool(doc.profile),
                    'has_search_hints':bool(doc.search_hints), 'has_embeddings':bool(embeddings)}
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
                _basename(sheet['csv_file'])
            doc.table_folder = TableFolder(self.client, self.bucket, root)
        if manifest.get('has_profile'):
            doc.profile = self._read_json(root + 'semantic_profile.json')
        if manifest.get('has_search_hints'):
            doc.search_hints = self._read_json(root + 'search_hints.json')
        if manifest.get('has_embeddings'):
            doc.embeddings = self._read_json(root + 'embeddings.json')
        doc.storage_ref = {'bucket':self.bucket, 'manifest_key':manifest_key}
        return doc

    def _remove_outputs(self, root):
        """Manifest first, so a half-removed document never appears complete; then the rest of its folder, then
        its vectors in the S3 Vectors index."""
        vector_keys = self._vector_keys(root)
        self.client.delete_object(Bucket=self.bucket, Key=root + 'manifest.json')
        pages = self.client.get_paginator('list_objects_v2').paginate(Bucket=self.bucket, Prefix=root)
        for key in [obj['Key'] for page in pages for obj in page.get('Contents', [])]:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        if vector_keys:
            self.vectors.delete(vector_keys)

    def _stored_embeddings(self, document_id, filename, embeddings):
        """What embeddings.json holds: the vectors themselves, or with an S3 Vectors index, the keys they were put
        under in this group's folder."""
        if self.vectors is None:
            return embeddings
        keys = self.vectors.put(self.prefix, document_id, filename, embeddings['vectors'])
        return {'meta': {**embeddings['meta'], 'vector_store': self.vectors.name, 'library': self.prefix,
                         'index': f'{self.vectors.bucket}/{self.vectors.index}'}, 'keys': keys}

    def _vector_keys(self, root):
        """S3 Vectors keys of the document saved at root; read before its files are removed."""
        if self.vectors is None:
            return []
        try:
            if not self._read_json(root + 'manifest.json').get('has_embeddings'):
                return []
            return self._read_json(root + 'embeddings.json').get('keys', [])
        except Exception:
            # Nothing saved there yet. Vectors left behind never match: search maps hits to current blocks.
            return []

    def delete(self, filename):
        """Remove a saved document (by filename or manifest key): its preprocessed folder, then its original.

        It leaves list_documents as soon as its manifest is removed. Deleting a missing document does nothing.
        """
        if self.read_only:
            raise PermissionError("This document store is read-only")
        with _trace('delete_document', filename=filename, group_name=self.group):
            root = self._root(self.manifest_key(filename))
            self._remove_outputs(root)
            self.client.delete_object(Bucket=self.bucket, Key=self._raw_key(root.split('/')[-2]))

    def save_profile(self, manifest_key, profile):
        """Store a generated or reviewed profile for an already saved document, replacing any earlier one."""
        if self.read_only:
            raise PermissionError("This document store is read-only")
        with _trace('save_profile', manifest_key=manifest_key, group_name=self.group) as current:
            root = self._root(manifest_key)
            manifest = self._read_json(manifest_key)
            if not isinstance(profile, dict) or not profile:
                raise ValueError('Profile must be a nonempty object')
            previous = self._read_json(root + 'semantic_profile.json') if manifest.get('has_profile') else {}
            self._json(root + 'semantic_profile.json', profile)
            self._json(manifest_key, {**manifest, 'has_profile': True})
            from backend.observability import tracing
            tracing.annotate(current, metadata=_profile_change(previous, profile))

    def save_search_hints(self, manifest_key, hints):
        """Store generated search hints for an already saved document, replacing any earlier ones."""
        if self.read_only:
            raise PermissionError("This document store is read-only")
        with _trace('save_search_hints', manifest_key=manifest_key, group_name=self.group):
            root = self._root(manifest_key)
            manifest = self._read_json(manifest_key)
            if not isinstance(hints, dict) or not isinstance(hints.get('blocks'), dict):
                raise ValueError('Search hints must be an object with blocks')
            self._json(root + 'search_hints.json', hints)
            self._json(manifest_key, {**manifest, 'has_search_hints': True})

    def save_embeddings(self, manifest_key, embeddings):
        """Store generated embeddings for an already saved document, replacing any earlier ones."""
        if self.read_only:
            raise PermissionError("This document store is read-only")
        with _trace('save_embeddings', manifest_key=manifest_key, group_name=self.group):
            root = self._root(manifest_key)
            manifest = self._read_json(manifest_key)
            if not isinstance(embeddings, dict) or not isinstance(embeddings.get('vectors'), dict):
                raise ValueError('Embeddings must be an object with vectors')
            previous = self._vector_keys(root)
            stored = self._stored_embeddings(manifest['document_id'], manifest['filename'], embeddings)
            self._json(root + 'embeddings.json', stored)
            self._json(manifest_key, {**manifest, 'has_embeddings': True})
            stale = sorted(set(previous) - set(stored.get('keys', [])))
            if stale:
                self.vectors.delete(stale)

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
