from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    start_layer: int
    end_layer: int  # exclusive
    includes_embed: bool
    includes_head: bool
    file_hashes: dict[str, str] = field(default_factory=dict)
    size_bytes: int = 0
    min_vram_mb: int = 0


@dataclass
class ModelManifest:
    model_id: str
    architecture: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    vocab_size: int
    max_seq_length: int
    dtype: str = "float16"
    quantization: str | None = None
    total_size_bytes: int = 0
    shards: list[ShardSpec] = field(default_factory=list)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> ModelManifest:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name_or_path)
        return cls(
            model_id=model_name_or_path,
            architecture=config.model_type,
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            vocab_size=config.vocab_size,
            max_seq_length=getattr(config, "max_position_embeddings", 4096),
        )

    def plan_shards(self, num_shards: int) -> list[ShardSpec]:
        layers_per_shard = self.num_layers // num_shards
        remainder = self.num_layers % num_shards
        shards = []
        offset = 0
        for i in range(num_shards):
            count = layers_per_shard + (1 if i < remainder else 0)
            end = offset + count
            shard = ShardSpec(
                shard_id=f"{self.model_id}:layers_{offset}_{end}",
                start_layer=offset,
                end_layer=end,
                includes_embed=(i == 0),
                includes_head=(i == num_shards - 1),
            )
            shards.append(shard)
            offset = end
        self.shards = shards
        return shards

    def to_dict(self) -> dict:
        return {
            "aitorrent_version": 1,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "vocab_size": self.vocab_size,
            "max_seq_length": self.max_seq_length,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "total_size_bytes": self.total_size_bytes,
            "shards": [
                {
                    "shard_id": s.shard_id,
                    "start_layer": s.start_layer,
                    "end_layer": s.end_layer,
                    "includes_embed": s.includes_embed,
                    "includes_head": s.includes_head,
                    "file_hashes": s.file_hashes,
                    "size_bytes": s.size_bytes,
                    "min_vram_mb": s.min_vram_mb,
                }
                for s in self.shards
            ],
        }

    def save(self, path: Path) -> None:
        data = self.to_dict()
        content = json.dumps(data, indent=2)
        data["manifest_sha256"] = hashlib.sha256(content.encode()).hexdigest()
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> ModelManifest:
        data = json.loads(path.read_text())
        shards = [
            ShardSpec(
                shard_id=s["shard_id"],
                start_layer=s["start_layer"],
                end_layer=s["end_layer"],
                includes_embed=s["includes_embed"],
                includes_head=s["includes_head"],
                file_hashes=s.get("file_hashes", {}),
                size_bytes=s.get("size_bytes", 0),
                min_vram_mb=s.get("min_vram_mb", 0),
            )
            for s in data["shards"]
        ]
        manifest = cls(
            model_id=data["model_id"],
            architecture=data["architecture"],
            num_layers=data["num_layers"],
            hidden_size=data["hidden_size"],
            num_attention_heads=data["num_attention_heads"],
            num_kv_heads=data["num_kv_heads"],
            vocab_size=data["vocab_size"],
            max_seq_length=data["max_seq_length"],
            dtype=data.get("dtype", "float16"),
            quantization=data.get("quantization"),
            total_size_bytes=data.get("total_size_bytes", 0),
            shards=shards,
        )
        return manifest
