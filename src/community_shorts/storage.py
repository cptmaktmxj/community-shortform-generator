"""Atomic JSON artifact storage for pipeline stage boundaries."""

import json
import os
import uuid
from pathlib import Path
from typing import Sequence, TypeVar

from pydantic import BaseModel

from community_shorts.models import CuratedItem, RawItem


ModelT = TypeVar("ModelT", bound=BaseModel)


class ArtifactStore:
    """Read and atomically replace Stage 1 and Stage 2 JSON artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.items_path = root / "items.json"
        self.curated_path = root / "curated.json"

    def read_items(self) -> list[RawItem]:
        """Read the current Stage 1 artifact, returning an empty list if absent."""

        return self._read_models(self.items_path, RawItem)

    def write_items(self, items: Sequence[RawItem]) -> None:
        """Merge items by ID and atomically replace the Stage 1 artifact."""

        merged = {item.item_id: item for item in self.read_items()}
        merged.update({item.item_id: item for item in items})
        self._write_models(self.items_path, [merged[key] for key in sorted(merged)])

    def read_curated(self) -> list[CuratedItem]:
        """Read the current Stage 2 artifact, returning an empty list if absent."""

        return self._read_models(self.curated_path, CuratedItem)

    def write_curated(self, items: Sequence[CuratedItem]) -> None:
        """Merge curated items by ID and atomically replace the Stage 2 artifact."""

        merged = {item.item_id: item for item in self.read_curated()}
        merged.update({item.item_id: item for item in items})
        self._write_models(self.curated_path, [merged[key] for key in sorted(merged)])

    def _read_models(self, path: Path, model_type: type[ModelT]) -> list[ModelT]:
        """Validate a JSON array against the requested artifact model."""

        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Artifact must contain a JSON array: {path}")
        return [model_type.model_validate(item) for item in payload]

    def _write_models(self, path: Path, items: Sequence[BaseModel]) -> None:
        """Flush a temporary file before replacing the artifact path."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    [item.model_dump(mode="json", by_alias=True) for item in items],
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
