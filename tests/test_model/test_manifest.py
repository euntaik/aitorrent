from pathlib import Path

import pytest

from aitorrent.model.manifest import ModelManifest, ShardSpec


@pytest.fixture
def manifest():
    return ModelManifest(
        model_id="test-model",
        architecture="llama",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        vocab_size=32000,
        max_seq_length=4096,
    )


def test_plan_shards_2(manifest):
    shards = manifest.plan_shards(2)
    assert len(shards) == 2
    assert shards[0].start_layer == 0
    assert shards[0].end_layer == 16
    assert shards[0].includes_embed is True
    assert shards[0].includes_head is False
    assert shards[1].start_layer == 16
    assert shards[1].end_layer == 32
    assert shards[1].includes_embed is False
    assert shards[1].includes_head is True


def test_plan_shards_3(manifest):
    shards = manifest.plan_shards(3)
    assert len(shards) == 3
    total = sum(s.end_layer - s.start_layer for s in shards)
    assert total == 32
    assert shards[0].start_layer == 0
    assert shards[-1].end_layer == 32


def test_save_and_load(manifest, tmp_path):
    manifest.plan_shards(2)
    path = tmp_path / "test.aitorrent"
    manifest.save(path)
    loaded = ModelManifest.load(path)
    assert loaded.model_id == manifest.model_id
    assert loaded.num_layers == manifest.num_layers
    assert len(loaded.shards) == 2


def test_to_dict(manifest):
    manifest.plan_shards(2)
    d = manifest.to_dict()
    assert d["aitorrent_version"] == 1
    assert d["model_id"] == "test-model"
    assert len(d["shards"]) == 2
