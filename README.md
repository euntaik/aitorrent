# AITorrent

**BitTorrent-inspired distributed LLM serving framework.**

Run models that are too large for any single machine by splitting them across connected peers. Peers that contribute compute earn credits, which they spend when requesting inference from others — the same reciprocity philosophy that made BitTorrent work, applied to AI model serving.

```
Client → Peer A (layers 0–19) → Peer B (layers 20–59) → Peer C (layers 60–79) → Client
         embed → hidden        → hidden                → hidden → logits
```

## How it works

- **Pipeline parallelism** — a transformer's layers are split into contiguous blocks, and each peer serves one block. For example, Llama-70B's 80 layers might be split across 3 peers based on each peer's VRAM and compute capability.
- **Forward pass chain** — hidden-state activation tensors flow peer-to-peer over gRPC. Autoregressive generation repeats the chain per token.
- **Credit economy** — every remote forward pass is paid for with credits (`base_rate × tokens × layers`). Transactions are Ed25519-signed and recorded in a local SQLite ledger with nonce-based replay protection. Tit-for-tat prioritization rewards peers that contribute.
- **Hardware-aware partitioning** — each peer profiles its GPU/RAM/TFLOPS at startup; layers are assigned proportionally to capacity.
- **Failover** — failed peers are detected mid-inference, blacklisted after repeated failures, and automatically replaced by backup peers covering the same layer range.
- **LAN discovery** — peers announce themselves via UDP broadcast so nodes can find each other without manual configuration.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/euntaik/aitorrent.git
cd aitorrent
pip install -e .

# with dev/test dependencies
pip install -e ".[dev]"

# with NVIDIA GPU profiling support
pip install -e ".[gpu]"
```

## Quickstart

### Single node (model fits locally)

```bash
aitorrent serve meta-llama/Llama-3.1-8B
```

### Two peers, model split in half

```bash
# Terminal 1 — Peer A serves layers 0–15 and runs the API
aitorrent serve meta-llama/Llama-3.1-8B --layers 0:16 --port 9877 --peer localhost:9878

# Terminal 2 — Peer B seeds layers 16–32
aitorrent seed meta-llama/Llama-3.1-8B --layers 16:32 --port 9878
```

### Send an inference request

The API is OpenAI-compatible, so any OpenAI SDK or tool works:

```bash
aitorrent infer "Explain quantum computing in simple terms"

# or stream tokens as they generate
aitorrent infer "Write a haiku about torrents" --stream

# or use plain HTTP
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B", "messages": [{"role": "user", "content": "Hello!"}]}'
```

See [examples/quickstart.py](examples/quickstart.py) for a Python client example.

## CLI reference

| Command | Description |
|---------|-------------|
| `aitorrent serve <model>` | Load model (or a layer range), join the network, start the OpenAI-compatible API |
| `aitorrent seed <model> --layers 16:32` | Serve a layer range without running the API endpoint |
| `aitorrent infer "<prompt>"` | Send an inference request to a running node (`--stream` for SSE) |
| `aitorrent peers` | Scan the LAN for peers announcing via UDP broadcast |
| `aitorrent balance` | Show credit balance and recent signed transactions |
| `aitorrent profile` | Benchmark local hardware and estimate model capacity |

Useful `serve` flags: `--peer host:port` (repeatable for multi-peer), `--discover` (enable LAN discovery), `--layers start:end`, `--device cpu|cuda`, `--api-port`.

## Architecture

```
aitorrent/
├── network/        # gRPC tensor transport, peer lifecycle, health checks, UDP discovery
├── model/          # hardware profiler, layer slicer, manifest, safetensors shard loader
├── inference/      # pipeline forward-pass chain, orchestrator, KV cache, scheduler, failover
├── credit/         # Ed25519 identity, signed ledger (SQLite), pricing, tit-for-tat
└── api/            # OpenAI-compatible FastAPI server + Typer CLI
```

Key design points:

- **Tensor transport**: activations are serialized with msgpack (bfloat16 → float16 on the wire) over gRPC with 256 MB message limits.
- **Identity**: each node has a persistent Ed25519 key pair (`~/.aitorrent/identity.pem`); the peer ID is derived from the public key.
- **Ledger**: bilateral local ledger — no blockchain. Signed transactions are exchanged and verified between peers; tampered or replayed transactions are rejected.
- **Verification**: random challenge tensors (5% probability) spot-check that peers actually compute what they claim.

## Development

```bash
pip install -e ".[dev]"
pytest            # 48 tests: unit + multi-peer integration over real gRPC
```

The integration tests build a tiny 4-layer transformer, split it across 2–3 in-process gRPC servers, and assert that distributed logits exactly match the single-node baseline — plus signed credit settlement across all peers.

## Roadmap

- **Phase 1 — MVP** ✅ 2-peer pipeline inference, gRPC transport, credit ledger, CLI
- **Phase 2 — Multi-peer & robustness** ✅ Ed25519 signed credits, N-peer orchestrator, failover, LAN discovery, 3-peer integration tests
- **Phase 3 — Production hardening** 🚧 NAT traversal, reputation system, GGUF quantized models, monitoring dashboard, `.aitorrent` model distribution format
- **Phase 4 — Advanced** speculative decoding across peers, tensor parallelism, model marketplace, web UI

## License

MIT
