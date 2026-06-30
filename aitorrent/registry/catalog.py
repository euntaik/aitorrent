from __future__ import annotations

from dataclasses import dataclass, field
from aitorrent.model.manifest import ModelManifest


@dataclass
class ModelEntry:
    manifest: ModelManifest
    providers: list[str] = field(default_factory=list)  # peer_ids
    shard_coverage: dict[int, list[str]] = field(default_factory=dict)  # layer_idx -> peer_ids


class ModelCatalog:
    def __init__(self):
        self._models: dict[str, ModelEntry] = {}

    def register(self, manifest: ModelManifest, peer_id: str) -> None:
        if manifest.model_id not in self._models:
            self._models[manifest.model_id] = ModelEntry(manifest=manifest)
        entry = self._models[manifest.model_id]
        if peer_id not in entry.providers:
            entry.providers.append(peer_id)

    def unregister(self, model_id: str, peer_id: str) -> None:
        entry = self._models.get(model_id)
        if entry and peer_id in entry.providers:
            entry.providers.remove(peer_id)

    def list_models(self) -> list[ModelManifest]:
        return [e.manifest for e in self._models.values() if e.providers]

    def get(self, model_id: str) -> ModelEntry | None:
        return self._models.get(model_id)

    def has_full_coverage(self, model_id: str) -> bool:
        entry = self._models.get(model_id)
        if not entry:
            return False
        return len(entry.providers) > 0
