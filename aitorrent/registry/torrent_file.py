from __future__ import annotations

from pathlib import Path
from aitorrent.model.manifest import ModelManifest


EXTENSION = ".aitorrent"


def create_torrent_file(manifest: ModelManifest, output_path: Path) -> Path:
    if not output_path.suffix:
        output_path = output_path.with_suffix(EXTENSION)
    manifest.save(output_path)
    return output_path


def load_torrent_file(path: Path) -> ModelManifest:
    return ModelManifest.load(path)
