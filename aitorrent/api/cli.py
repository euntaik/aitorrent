from __future__ import annotations

import asyncio
import logging

import typer
from rich.console import Console
from rich.table import Table

from aitorrent.config import AITorrentConfig

app = typer.Typer(name="aitorrent", help="Distributed AI model serving framework")
console = Console()


@app.command()
def serve(
    model: str = typer.Argument(..., help="Model ID (e.g., meta-llama/Llama-3.1-8B)"),
    peer_address: str = typer.Option(None, "--peer", "-p", help="Remote peer address (host:port)"),
    port: int = typer.Option(9877, "--port", help="gRPC listen port"),
    api_port: int = typer.Option(8000, "--api-port", help="API server port"),
    layers: str = typer.Option(None, "--layers", "-l", help="Layer range (e.g., 0:16)"),
    device: str = typer.Option("auto", "--device", "-d", help="Device (auto/cpu/cuda)"),
):
    """Start serving model shards and join the network."""
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    config = AITorrentConfig()
    config.network.grpc_port = port

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    asyncio.run(_serve_async(config, model, peer_address, api_port, layers, device))


async def _serve_async(
    config: AITorrentConfig,
    model_id: str,
    peer_address: str | None,
    api_port: int,
    layer_range: str | None,
    device: str,
):
    import torch
    from aitorrent.model.manifest import ModelManifest
    from aitorrent.model.loader import ShardLoader
    from aitorrent.model.slicer import ModelSlicer
    from aitorrent.network.peer import PeerNode, PeerInfo
    from aitorrent.inference.pipeline import InferencePipeline, PipelineStage
    from aitorrent.credit.ledger import CreditLedger
    from aitorrent.credit.pricing import CreditPricer

    node = PeerNode(config)
    await node.start()

    console.print(f"[bold green]Node started:[/] {node.peer_id}")
    console.print(f"[dim]Device: {device} | gRPC port: {config.network.grpc_port}[/]")

    console.print(f"Loading model [bold]{model_id}[/]...")
    manifest = ModelManifest.from_pretrained(model_id)

    if peer_address:
        # 2-peer mode: split model between self and remote peer
        slicer = ModelSlicer()
        peer_info = PeerInfo(peer_id="remote", address=peer_address)
        conn = await node.connect_to_peer(peer_info)

        if layer_range:
            start, end = map(int, layer_range.split(":"))
        else:
            mid = manifest.num_layers // 2
            start, end = 0, mid

        console.print(f"Loading layers {start}-{end} locally...")
        loader = ShardLoader()
        dtype = torch.float16 if device == "cuda" else torch.float32
        local_shard = loader.load_from_pretrained(
            model_id, manifest, start, end,
            includes_embed=(start == 0),
            includes_head=(end == manifest.num_layers),
            device=device, dtype=dtype,
        )
        node.load_shard(model_id, local_shard)

        remote_info = PeerInfo(
            peer_id="remote", address=peer_address,
            start_layer=end, end_layer=manifest.num_layers,
            includes_embed=False, includes_head=True,
        )
        stages = [
            PipelineStage(
                peer_info=PeerInfo(peer_id=node.peer_id, address=node.address),
                connection=None,
                local_shard=local_shard,
            ),
            PipelineStage(
                peer_info=remote_info,
                connection=conn,
                local_shard=None,
            ),
        ]
    else:
        # Single node mode: load full model
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
                peer_info=PeerInfo(peer_id=node.peer_id, address=node.address),
                connection=None,
                local_shard=local_shard,
            ),
        ]

    ledger = CreditLedger(node.peer_id, config.credit.db_path)
    pricer = CreditPricer()
    pipeline = InferencePipeline(node, manifest, stages, ledger, pricer)

    # Start API server
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
        await node.stop()


@app.command()
def infer(
    prompt: str = typer.Argument(..., help="Prompt text"),
    model: str = typer.Option("meta-llama/Llama-3.1-8B", "--model", "-m"),
    api_url: str = typer.Option("http://localhost:8000", "--api-url"),
    max_tokens: int = typer.Option(256, "--max-tokens"),
    temperature: float = typer.Option(0.7, "--temperature"),
):
    """Send an inference request to a running AITorrent node."""
    import httpx

    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{api_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        console.print(f"\n[bold]Response:[/]\n{text}")
        console.print(f"\n[dim]Tokens: {data['usage']['total_tokens']}[/]")


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
        for tx in txns:
            import datetime
            ts = datetime.datetime.fromtimestamp(tx.timestamp).strftime("%H:%M:%S")
            tx_table.add_row(ts, tx.from_peer[:8], tx.to_peer[:8], f"{tx.amount:.2f}", tx.reason)
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
    console.print(table)


@app.command()
def peers(
    api_url: str = typer.Option("http://localhost:8000", "--api-url"),
):
    """List connected peers."""
    console.print("[dim]Peer discovery not yet implemented in MVP. Use --peer flag with serve.[/]")


if __name__ == "__main__":
    app()
