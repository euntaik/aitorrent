from __future__ import annotations

from dataclasses import dataclass

from aitorrent.model.manifest import ModelManifest
from aitorrent.model.profiler import HardwareProfile
from aitorrent.model.slicer import LayerAssignment, ModelSlicer


@dataclass
class PeerCapability:
    peer_id: str
    address: str
    profile: HardwareProfile
    models: list[str]
    assigned_layers: LayerAssignment | None = None


def assign_peers(
    manifest: ModelManifest,
    peers: list[PeerCapability],
) -> list[PeerCapability]:
    slicer = ModelSlicer()
    profiles = [(p.peer_id, p.profile) for p in peers]
    assignments = slicer.assign_layers(manifest, profiles)
    for peer, assignment in zip(peers, assignments):
        peer.assigned_layers = assignment
    return peers
