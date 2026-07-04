from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from aitorrent.model.manifest import ModelManifest

logger = logging.getLogger(__name__)


@dataclass
class TransformerShard:
    model_id: str
    start_layer: int
    end_layer: int
    layers: nn.ModuleList
    embed: nn.Module | None
    head: nn.Linear | None
    norm: nn.Module | None
    device: str
    dtype: torch.dtype
    rotary_emb: nn.Module | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        new_cache = {} if kv_cache is not None else None

        if position_ids is None:
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(
                seq_len, device=hidden_states.device
            ).unsqueeze(0)

        # Modern HF layers expect (cos, sin) computed at the model level
        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, layer in enumerate(self.layers):
            layer_idx = self.start_layer + i
            past_kv = kv_cache.get(layer_idx) if kv_cache else None

            if hasattr(layer, "self_attn"):
                # HuggingFace-style layer
                kwargs = dict(
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=kv_cache is not None,
                )
                if position_embeddings is not None:
                    kwargs["position_embeddings"] = position_embeddings
                outputs = layer(hidden_states, **kwargs)
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
                if new_cache is not None and isinstance(outputs, tuple) and len(outputs) > 1:
                    new_cache[layer_idx] = outputs[1]
            else:
                hidden_states = layer(hidden_states)

        return hidden_states, new_cache


class ShardLoader:
    def load_from_pretrained(
        self,
        model_name_or_path: str,
        manifest: ModelManifest,
        start_layer: int,
        end_layer: int,
        includes_embed: bool = False,
        includes_head: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> TransformerShard:
        from transformers import AutoModelForCausalLM

        logger.info(
            "Loading layers %d-%d from %s", start_layer, end_layer, model_name_or_path
        )
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
        )

        base = self._get_model_base(full_model)
        layer_list = self._get_layers(base)

        shard_layers = nn.ModuleList(
            [layer_list[i] for i in range(start_layer, end_layer)]
        )

        embed = None
        if includes_embed:
            embed = self._get_embed(base)

        head = None
        norm = None
        if includes_head:
            head = self._get_head(full_model)
            norm = self._get_final_norm(base)

        shard = TransformerShard(
            model_id=manifest.model_id,
            start_layer=start_layer,
            end_layer=end_layer,
            layers=shard_layers,
            embed=embed,
            head=head,
            norm=norm,
            device=device,
            dtype=dtype,
            rotary_emb=getattr(base, "rotary_emb", None),
        )

        # Move only the shard to target device, delete full model
        for component in [shard.layers, shard.embed, shard.head, shard.norm, shard.rotary_emb]:
            if component is not None:
                component.to(device)
        del full_model
        if device == "cuda":
            torch.cuda.empty_cache()

        logger.info("Shard loaded: layers %d-%d on %s", start_layer, end_layer, device)
        return shard

    def load_from_safetensors(
        self,
        shard_path: Path,
        manifest: ModelManifest,
        start_layer: int,
        end_layer: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> dict[str, torch.Tensor]:
        tensors = load_file(str(shard_path), device=device)
        return {k: v.to(dtype) for k, v in tensors.items()}

    def _get_model_base(self, model: nn.Module) -> nn.Module:
        for attr in ["model", "transformer", "gpt_neox"]:
            if hasattr(model, attr):
                return getattr(model, attr)
        raise ValueError(f"Unknown model architecture: {type(model)}")

    def _get_layers(self, base: nn.Module) -> nn.ModuleList:
        for attr in ["layers", "h", "block"]:
            if hasattr(base, attr):
                return getattr(base, attr)
        raise ValueError(f"Cannot find layer list in {type(base)}")

    def _get_embed(self, base: nn.Module) -> nn.Module | None:
        for attr in ["embed_tokens", "wte", "word_embeddings"]:
            if hasattr(base, attr):
                return getattr(base, attr)
        return None

    def _get_head(self, model: nn.Module) -> nn.Module | None:
        for attr in ["lm_head", "output"]:
            if hasattr(model, attr):
                return getattr(model, attr)
        return None

    def _get_final_norm(self, base: nn.Module) -> nn.Module | None:
        for attr in ["norm", "ln_f", "final_layer_norm"]:
            if hasattr(base, attr):
                return getattr(base, attr)
        return None
