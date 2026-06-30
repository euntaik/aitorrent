from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from aitorrent.model.manifest import ModelManifest, ShardSpec
from aitorrent.model.profiler import HardwareProfile

logger = logging.getLogger(__name__)


@dataclass
class LayerAssignment:
    peer_id: str
    start_layer: int
    end_layer: int
    includes_embed: bool
    includes_head: bool
    estimated_vram_mb: int


class ModelSlicer:
    def assign_layers(
        self,
        manifest: ModelManifest,
        peer_profiles: list[tuple[str, HardwareProfile]],
    ) -> list[LayerAssignment]:
        dtype_bytes = 2 if manifest.dtype in ("float16", "bfloat16") else 4
        params_per_layer = self._estimate_params_per_layer(manifest)

        capacities = []
        for peer_id, profile in peer_profiles:
            max_l = profile.max_layers(params_per_layer, dtype_bytes)
            capacities.append((peer_id, max(max_l, 1), profile))

        total_capacity = sum(c for _, c, _ in capacities)
        assignments = []
        offset = 0

        for i, (peer_id, cap, profile) in enumerate(capacities):
            is_last = i == len(capacities) - 1
            if is_last:
                count = manifest.num_layers - offset
            else:
                ratio = cap / total_capacity
                count = max(1, round(manifest.num_layers * ratio))
                count = min(count, manifest.num_layers - offset)

            end = offset + count
            layer_mb = (params_per_layer * dtype_bytes * count) / (1024 * 1024)

            assignments.append(
                LayerAssignment(
                    peer_id=peer_id,
                    start_layer=offset,
                    end_layer=end,
                    includes_embed=(i == 0),
                    includes_head=is_last,
                    estimated_vram_mb=int(layer_mb),
                )
            )
            offset = end
            if offset >= manifest.num_layers:
                break

        return assignments

    def split_safetensors(
        self,
        model_path: Path,
        manifest: ModelManifest,
        assignments: list[LayerAssignment],
        output_dir: Path,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        all_tensors = {}
        for sf_file in sorted(model_path.glob("*.safetensors")):
            all_tensors.update(load_file(str(sf_file)))

        shard_paths = []
        for assignment in assignments:
            shard_tensors = {}
            for key, tensor in all_tensors.items():
                if self._key_belongs_to_assignment(key, assignment, manifest):
                    shard_tensors[key] = tensor

            shard_name = f"shard_{assignment.start_layer}_{assignment.end_layer}.safetensors"
            shard_path = output_dir / shard_name
            save_file(shard_tensors, str(shard_path))
            shard_paths.append(shard_path)
            logger.info(
                "Saved shard layers %d-%d (%d tensors) to %s",
                assignment.start_layer,
                assignment.end_layer,
                len(shard_tensors),
                shard_path,
            )

        return shard_paths

    def _key_belongs_to_assignment(
        self, key: str, assignment: LayerAssignment, manifest: ModelManifest
    ) -> bool:
        if "layers." in key or "h." in key:
            layer_idx = self._extract_layer_index(key)
            if layer_idx is not None:
                return assignment.start_layer <= layer_idx < assignment.end_layer
            return False

        if assignment.includes_embed and self._is_embed_key(key):
            return True
        if assignment.includes_head and self._is_head_key(key):
            return True

        return False

    def _extract_layer_index(self, key: str) -> int | None:
        for part in key.split("."):
            if part.isdigit():
                return int(part)
        return None

    def _is_embed_key(self, key: str) -> bool:
        embed_patterns = ["embed_tokens", "wte", "word_embeddings", "embed_in"]
        return any(p in key for p in embed_patterns)

    def _is_head_key(self, key: str) -> bool:
        head_patterns = ["lm_head", "output", "ln_f", "norm", "final_layer_norm"]
        return any(p in key for p in head_patterns)

    def _estimate_params_per_layer(self, manifest: ModelManifest) -> int:
        h = manifest.hidden_size
        kv_h = manifest.num_kv_heads
        n_h = manifest.num_attention_heads
        head_dim = h // n_h
        # Q, K, V projections + output projection + MLP (gate, up, down with 8/3 ratio)
        attn_params = h * (n_h * head_dim + 2 * kv_h * head_dim + h)
        ffn_dim = int(h * 8 / 3)
        ffn_params = h * ffn_dim * 3  # gate, up, down
        norm_params = h * 2  # two layer norms
        return attn_params + ffn_params + norm_params
