"""Document records shared by ingestion, storage and agents."""
from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    name: str
    kind: str
    blocks: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    table_folder: object | None = None  # backend.storage.tables: TableFolder or MemoryTableFolder
    table_metadata: dict | None = None

    storage_ref: dict | None = None
    profile: dict | None = None
    search_hints: dict | None = None  # backend/agents/search_hints.py; routes search, never evidence
    embeddings: dict | None = None  # backend/agents/embeddings.py; routes semantic search, never evidence

    def summary(self):
        result = {'document_id':self.id, 'filename':self.name, 'type':self.kind,
                  'text_blocks':len(self.blocks), 'warnings':self.warnings}
        if self.table_metadata:
            from backend.shared.metadata import sheet_index
            result['sheets'] = sheet_index(self.table_metadata)
        if self.profile:
            from backend.shared.profile import profile_hints
            result['semantic_profile'] = profile_hints(self.profile)
        return result
