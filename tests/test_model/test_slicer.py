import pytest

from aitorrent.model.manifest import ModelManifest
from aitorrent.model.profiler import HardwareProfile
from aitorrent.model.slicer import ModelSlicer


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


def _make_profile(vram_mb: int, ram_mb: int) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="Test GPU",
        gpu_vram_total_mb=vram_mb,
        gpu_vram_free_mb=vram_mb,
        ram_total_mb=ram_mb,
        ram_free_mb=ram_mb,
        cpu_cores=8,
        compute_tflops=10.0,
    )


def test_assign_equal_peers(manifest):
    slicer = ModelSlicer()
    peers = [
        ("peer_a", _make_profile(8000, 32000)),
        ("peer_b", _make_profile(8000, 32000)),
    ]
    assignments = slicer.assign_layers(manifest, peers)
    assert len(assignments) == 2
    total = sum(a.end_layer - a.start_layer for a in assignments)
    assert total == 32
    assert assignments[0].includes_embed is True
    assert assignments[-1].includes_head is True


def test_assign_unequal_peers(manifest):
    slicer = ModelSlicer()
    peers = [
        ("peer_a", _make_profile(16000, 32000)),
        ("peer_b", _make_profile(4000, 16000)),
    ]
    assignments = slicer.assign_layers(manifest, peers)
    assert len(assignments) == 2
    # Peer A should get more layers
    a_layers = assignments[0].end_layer - assignments[0].start_layer
    b_layers = assignments[1].end_layer - assignments[1].start_layer
    assert a_layers + b_layers == 32
    assert a_layers > b_layers
