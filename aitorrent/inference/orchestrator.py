"""Build and manage a distributed inference pipeline from available peers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from aitorrent.credit.ledger import CreditLedger
from aitorrent.credit.pricing import CreditPricer
from aitorrent.inference.pipeline import InferencePipeline, PipelineStage
from aitorrent.model.loader import ShardLoader, TransformerShard
from aitorrent.model.manifest import ModelManifest
from aitorrent.model.slicer import ModelSlicer
from aitorrent.network.peer import PeerInfo, PeerNode
from aitorrent.network.transport import PeerConnection

logger = logging.getLogger(__name__)


@dataclass
class PeerSlot:
    peer_info: PeerInfo
    connection: PeerConnection | None
    local_shard: TransformerShard | None
    start_layer: int
    end_layer: int
    includes_embed: bool
    includes_head: bool


class PipelineOrchestrator:
    """Assigns layers to peers and builds a ready-to-use pipeline."""

    def __init__(
        self,
        node: PeerNode,
        manifest: ModelManifest,
        ledger: CreditLedger,
        pricer: CreditPricer | None = None,
    ):
        self._node = node
        self._manifest = manifest
        self._ledger = ledger
        self._pricer = pricer or CreditPricer()
        self._slots: list[PeerSlot] = []

    async def build_pipeline(
        self,
        remote_peers: list[PeerInfo],
        local_layers: tuple[int, int] | None = None,
        model_name_or_path: str | None = None,
        device: str = "cpu",
    ) -> InferencePipeline:
        """Build a pipeline across self + remote peers.

        If local_layers is not given, auto-assign layers based on hardware.
        """
        import torch

        model_path = model_name_or_path or self._manifest.model_id
        num_layers = self._manifest.num_layers
        all_peers = self._collect_peers(remote_peers)
        assignments = self._assign_layers(all_peers, local_layers)

        stages = []
        for asgn in assignments:
            if asgn.peer_info.peer_id == self._node.peer_id:
                # Local: load shard
                dtype = torch.float16 if device == "cuda" else torch.float32
                loader = ShardLoader()
                shard = loader.load_from_pretrained(
                    model_path,
                    self._manifest,
                    asgn.start_layer,
                    asgn.end_layer,
                    includes_embed=asgn.includes_embed,
                    includes_head=asgn.includes_head,
                    device=device,
                    dtype=dtype,
                )
                self._node.load_shard(self._manifest.model_id, shard)
                stages.append(PipelineStage(
                    peer_info=asgn.peer_info,
                    connection=None,
                    local_shard=shard,
                ))
            else:
                # Remote: connect
                conn = self._node.get_connection(asgn.peer_info.peer_id)
                if conn is None:
                    conn = await self._node.connect_to_peer(asgn.peer_info)
                stages.append(PipelineStage(
                    peer_info=asgn.peer_info,
                    connection=conn,
                    local_shard=None,
                ))

        logger.info(
            "Pipeline built: %d stages covering %d layers",
            len(stages), num_layers,
        )
        return InferencePipeline(
            self._node, self._manifest, stages, self._ledger, self._pricer,
        )

    def _collect_peers(self, remote_peers: list[PeerInfo]) -> list[PeerInfo]:
        local_peer = self._node.peer_info()
        return [local_peer] + remote_peers

    def _assign_layers(
        self,
        peers: list[PeerInfo],
        local_layers: tuple[int, int] | None,
    ) -> list[PeerSlot]:
        num_layers = self._manifest.num_layers

        if local_layers is not None and len(peers) == 2:
            # Manual 2-peer split — also record ranges on peer_info so that
            # credit pricing sees the correct layer counts
            local = peers[0]
            remote = peers[1]
            local.start_layer, local.end_layer = local_layers
            local.includes_embed = local_layers[0] == 0
            local.includes_head = local_layers[1] == num_layers
            remote.start_layer, remote.end_layer = local_layers[1], num_layers
            remote.includes_embed = local_layers[1] == 0
            remote.includes_head = True
            return [
                PeerSlot(
                    peer_info=local, connection=None, local_shard=None,
                    start_layer=local.start_layer, end_layer=local.end_layer,
                    includes_embed=local.includes_embed,
                    includes_head=local.includes_head,
                ),
                PeerSlot(
                    peer_info=remote, connection=None, local_shard=None,
                    start_layer=remote.start_layer, end_layer=remote.end_layer,
                    includes_embed=remote.includes_embed,
                    includes_head=remote.includes_head,
                ),
            ]

        # Auto-assign based on hardware capability
        slicer = ModelSlicer()
        profiles = []
        for p in peers:
            if p.profile is None:
                from aitorrent.model.profiler import HardwareProfile
                profile = HardwareProfile(
                    gpu_name=None, gpu_vram_total_mb=0, gpu_vram_free_mb=0,
                    ram_total_mb=8000, ram_free_mb=4000,
                    cpu_cores=4, compute_tflops=0.1,
                )
            else:
                profile = p.profile
            profiles.append((p.peer_id, profile))

        assignments = slicer.assign_layers(self._manifest, profiles)

        slots = []
        for peer, asgn in zip(peers, assignments):
            slots.append(PeerSlot(
                peer_info=peer, connection=None, local_shard=None,
                start_layer=asgn.start_layer, end_layer=asgn.end_layer,
                includes_embed=asgn.includes_embed,
                includes_head=asgn.includes_head,
            ))
            # Update peer_info with assignment
            peer.start_layer = asgn.start_layer
            peer.end_layer = asgn.end_layer
            peer.includes_embed = asgn.includes_embed
            peer.includes_head = asgn.includes_head

        return slots
