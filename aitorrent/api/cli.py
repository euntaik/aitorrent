from __future__ import annotations

import asyncio
import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from aitorrent.config import AITorrentConfig

app = typer.Typer(name="aitorrent", help="Distributed AI model serving framework")
console = Console()


@app.command()
def serve(
    model: str = typer.Argument(..., help="Model ID (e.g., meta-llama/Llama-3.1-8B)"),
    peer_address: list[str] = typer.Option([], "--peer", "-p", help="Remote peer address(es)"),
    port: int = typer.Option(9877, "--port", help="gRPC listen port"),
    api_port: int = typer.Option(8000, "--api-port", help="API server port"),
    layers: str = typer.Option(None, "--layers", "-l", help="Layer range (e.g., 0:16)"),
    device: str = typer.Option("auto", "--device", "-d", help="Device (auto/cpu/cuda)"),
    discover: bool = typer.Option(False, "--discover", help="Enable LAN peer discovery"),
    discover_port: int = typer.Option(9876, "--discover-port", help="UDP broadcast port"),
):
    """Start serving model shards and join the network."""
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    config = AITorrentConfig()
    config.network.grpc_port = port

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    asyncio.run(_serve_async(
        config, model, peer_address, api_port, layers, device,
        discover, discover_port,
    ))


async def _serve_async(
    config: AITorrentConfig,
    model_id: str,
    peer_addresses: list[str],
    api_port: int,
    layer_range: str | None,
    device: str,
    discover: bool,
    discover_port: int,
):
    import torch
    from aitorrent.credit.crypto import PeerIdentity
    from aitorrent.credit.ledger import CreditLedger
    from aitorrent.credit.pricing import CreditPricer
    from aitorrent.inference.failover import FailoverManager
    from aitorrent.inference.pipeline import InferencePipeline, PipelineStage
    from aitorrent.model.loader import ShardLoader
    from aitorrent.model.manifest import ModelManifest
    from aitorrent.network.peer import PeerInfo, PeerNode

    node = PeerNode(config)
    await node.start()

    console.print(f"[bold green]Node started:[/] {node.peer_id[:16]}...")
    console.print(f"[dim]Device: {device} | gRPC port: {config.network.grpc_port}[/]")

    dht = None
    if discover:
        from aitorrent.network.discovery import DHTDiscovery
        dht = DHTDiscovery(
            local_peer=node.peer_info(),
            broadcast_port=discover_port,
        )
        dht.register_model(model_id)
        await dht.start()
        console.print(f"[bold cyan]LAN discovery enabled[/] on port {discover_port}")

    console.print(f"Loading model [bold]{model_id}[/]...")
    manifest = ModelManifest.from_pretrained(model_id)

    if peer_addresses:
        from aitorrent.inference.orchestrator import PipelineOrchestrator

        remote_peers = []
        for i, addr in enumerate(peer_addresses):
            peer_info = PeerInfo(peer_id=f"remote_{i}", address=addr)
            await node.connect_to_peer(peer_info)
            remote_peers.append(peer_info)
            console.print(f"  Connected to peer [bold]{addr}[/]")

        local_layers = None
        if layer_range:
            start, end = map(int, layer_range.split(":"))
            local_layers = (start, end)

        ledger = CreditLedger(
            node.peer_id, config.credit.db_path,
            identity=node.identity,
        )
        pricer = CreditPricer()
        failover = FailoverManager()
        failover.register_backups(model_id, remote_peers)

        orchestrator = PipelineOrchestrator(node, manifest, ledger, pricer)
        pipeline = await orchestrator.build_pipeline(
            remote_peers, local_layers, model_id, device,
        )
        pipeline._failover = failover
    else:
        console.print("Loading full model locally...")
        loader = ShardLoader()
        dtype = torch.float16 if device == "cuda" else torch.float32
        local_shard = loader.load_from_pretrained(
            model_id, manifest, 0, manifest.num_layers,
            includes_embed=True, includes_head=True,
            device=device, dtype=dtype,
        )
        node.load_shard(model_id, local_shard)
        stages = [
            PipelineStage(
                peer_info=node.peer_info(),
                connection=None,
                local_shard=local_shard,
            ),
        ]
        ledger = CreditLedger(
            node.peer_id, config.credit.db_path,
            identity=node.identity,
        )
        pricer = CreditPricer()
        pipeline = InferencePipeline(node, manifest, stages, ledger, pricer)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    from aitorrent.api.server import app as api_app, configure
    configure(pipeline, manifest, ledger, tokenizer)

    import uvicorn
    console.print(f"[bold green]API server starting on port {api_port}[/]")
    console.print(f"[dim]OpenAI-compatible endpoint: http://localhost:{api_port}/v1/chat/completions[/]")

    server_config = uvicorn.Config(api_app, host="0.0.0.0", port=api_port, log_level="info")
    server = uvicorn.Server(server_config)
    try:
        await server.serve()
    finally:
        if dht:
            await dht.stop()
        await node.stop()


@app.command()
def infer(
    prompt: str = typer.Argument(..., help="Prompt text"),
    model: str = typer.Option("meta-llama/Llama-3.1-8B", "--model", "-m"),
    api_url: str = typer.Option("http://localhost:8000", "--api-url"),
    max_tokens: int = typer.Option(256, "--max-tokens"),
    temperature: float = typer.Option(0.7, "--temperature"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream response tokens"),
):
    """Send an inference request to a running AITorrent node."""
    import httpx

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    if stream:
        with httpx.Client(timeout=120) as client:
            with client.stream(
                "POST", f"{api_url}/v1/chat/completions", json=payload,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        import json
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            console.print(delta["content"], end="")
                console.print()
    else:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{api_url}/v1/chat/completions", json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            console.print(f"\n[bold]Response:[/]\n{text}")
            console.print(f"\n[dim]Tokens: {data['usage']['total_tokens']}[/]")


@app.command()
def seed(
    model: str = typer.Argument(..., help="Model ID to seed (e.g., meta-llama/Llama-3.1-8B)"),
    port: int = typer.Option(9877, "--port", help="gRPC listen port"),
    layers: str = typer.Option(None, "--layers", "-l", help="Layer range (e.g., 16:32)"),
    device: str = typer.Option("auto", "--device", "-d", help="Device"),
    discover: bool = typer.Option(True, "--discover/--no-discover", help="Enable LAN discovery"),
):
    """Seed model shards — serve layers without running the API endpoint."""
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    config = AITorrentConfig()
    config.network.grpc_port = port

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    asyncio.run(_seed_async(config, model, layers, device, discover))


async def _seed_async(
    config: AITorrentConfig,
    model_id: str,
    layer_range: str | None,
    device: str,
    discover: bool,
):
    import torch
    from aitorrent.model.loader import ShardLoader
    from aitorrent.model.manifest import ModelManifest
    from aitorrent.network.peer import PeerNode

    node = PeerNode(config)
    await node.start()

    console.print(f"[bold green]Seed node started:[/] {node.peer_id[:16]}...")

    manifest = ModelManifest.from_pretrained(model_id)

    if layer_range:
        start, end = map(int, layer_range.split(":"))
    else:
        start, end = 0, manifest.num_layers

    console.print(f"Loading layers {start}-{end} of [bold]{model_id}[/]...")
    loader = ShardLoader()
    dtype = torch.float16 if device == "cuda" else torch.float32
    shard = loader.load_from_pretrained(
        model_id, manifest, start, end,
        includes_embed=(start == 0),
        includes_head=(end == manifest.num_layers),
        device=device, dtype=dtype,
    )
    node.load_shard(model_id, shard)

    dht = None
    if discover:
        from aitorrent.network.discovery import DHTDiscovery
        dht = DHTDiscovery(
            local_peer=node.peer_info(),
            broadcast_port=9876,
        )
        dht.register_model(model_id)
        await dht.start()
        console.print("[bold cyan]LAN discovery enabled[/]")

    console.print(f"[bold green]Seeding {end - start} layers on port {config.network.grpc_port}[/]")
    console.print("[dim]Press Ctrl+C to stop[/]")

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        if dht:
            await dht.stop()
        await node.stop()


@app.command()
def balance(
    data_dir: str = typer.Option("~/.aitorrent", "--data-dir"),
):
    """Show credit balance."""
    from pathlib import Path
    from aitorrent.credit.ledger import CreditLedger

    db_path = Path(data_dir).expanduser() / "credits.db"
    if not db_path.exists():
        console.print("[yellow]No credit database found. Run 'aitorrent serve' first.[/]")
        raise typer.Exit(1)

    ledger = CreditLedger("local", db_path, bootstrap_credits=0)
    table = Table(title="Credit Balance")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Current Balance", f"{ledger.balance:.2f}")
    table.add_row("Total Earned", f"{ledger.total_earned():.2f}")
    table.add_row("Total Spent", f"{ledger.total_spent():.2f}")
    console.print(table)

    txns = ledger.recent_transactions(10)
    if txns:
        tx_table = Table(title="Recent Transactions")
        tx_table.add_column("Time")
        tx_table.add_column("From")
        tx_table.add_column("To")
        tx_table.add_column("Amount", justify="right")
        tx_table.add_column("Reason")
        tx_table.add_column("Signed", justify="center")
        for tx in txns:
            import datetime
            ts = datetime.datetime.fromtimestamp(tx.timestamp).strftime("%H:%M:%S")
            signed = "[green]Y[/]" if tx.signature else "[red]N[/]"
            tx_table.add_row(
                ts, tx.from_peer[:8], tx.to_peer[:8],
                f"{tx.amount:.4f}", tx.reason, signed,
            )
        console.print(tx_table)


@app.command()
def profile():
    """Run hardware benchmark and show results."""
    from aitorrent.model.profiler import HardwareProfiler

    console.print("[bold]Running hardware benchmark...[/]")
    profiler = HardwareProfiler()
    hw = profiler.profile()

    table = Table(title="Hardware Profile")
    table.add_column("Property", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("GPU", hw.gpu_name or "None")
    table.add_row("VRAM Total", f"{hw.gpu_vram_total_mb} MB")
    table.add_row("VRAM Free", f"{hw.gpu_vram_free_mb} MB")
    table.add_row("RAM Total", f"{hw.ram_total_mb} MB")
    table.add_row("RAM Free", f"{hw.ram_free_mb} MB")
    table.add_row("CPU Cores", str(hw.cpu_cores))
    table.add_row("Compute", f"{hw.compute_tflops:.2f} TFLOPS")

    from aitorrent.model.manifest import ModelManifest
    ref_models = [
        ("Llama-3.1-8B", 32, 4096, 14_000),
        ("Llama-3.1-70B", 80, 8192, 140_000),
        ("Llama-3.1-405B", 126, 16384, 810_000),
    ]
    table.add_section()
    for name, n_layers, hidden, mb_per_model in ref_models:
        mb_per_layer = mb_per_model / n_layers
        can_fit = int(hw.gpu_vram_free_mb / mb_per_layer) if hw.gpu_vram_free_mb > 0 else int(hw.ram_free_mb / mb_per_layer)
        pct = min(100, int(can_fit / n_layers * 100))
        table.add_row(f"{name} capacity", f"{can_fit}/{n_layers} layers ({pct}%)")

    console.print(table)


@app.command()
def peers(
    port: int = typer.Option(9876, "--port", help="Discovery broadcast port"),
    wait: int = typer.Option(5, "--wait", "-w", help="Seconds to listen for peers"),
):
    """Discover peers on the local network."""
    asyncio.run(_peers_async(port, wait))


async def _peers_async(port: int, wait: int):
    from aitorrent.network.discovery import DHTDiscovery
    from aitorrent.network.peer import PeerInfo

    local = PeerInfo(peer_id="scanner", address="localhost:0")
    dht = DHTDiscovery(local_peer=local, broadcast_port=port)
    await dht.start()

    console.print(f"[dim]Listening for peer announcements for {wait}s...[/]")
    await asyncio.sleep(wait)

    found = dht.all_peers()
    await dht.stop()

    if not found:
        console.print("[yellow]No peers found on the local network.[/]")
        console.print("[dim]Peers must be running with --discover flag.[/]")
        return

    table = Table(title=f"Discovered Peers ({len(found)})")
    table.add_column("Peer ID")
    table.add_column("Address")
    table.add_column("Public Key")
    for p in found:
        table.add_row(
            p.peer_id[:16] + "...",
            p.address,
            p.pubkey.hex()[:16] + "..." if p.pubkey else "N/A",
        )
    console.print(table)


if __name__ == "__main__":
    app()
