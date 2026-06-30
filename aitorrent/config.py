from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NetworkConfig:
    listen_host: str = "0.0.0.0"
    listen_port: int = 9876
    grpc_port: int = 9877
    bootstrap_peers: list[str] = field(default_factory=list)
    heartbeat_interval_sec: int = 10
    peer_timeout_sec: int = 30


@dataclass
class InferenceConfig:
    max_batch_size: int = 8
    max_seq_length: int = 4096
    kv_cache_ttl_sec: int = 300
    forward_timeout_sec: float = 30.0


@dataclass
class CreditConfig:
    bootstrap_credits: int = 1000
    base_rate_per_token_per_layer: float = 0.01
    db_path: Path = Path("~/.aitorrent/credits.db")


@dataclass
class AITorrentConfig:
    data_dir: Path = field(default_factory=lambda: Path("~/.aitorrent").expanduser())
    network: NetworkConfig = field(default_factory=NetworkConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    credit: CreditConfig = field(default_factory=CreditConfig)

    def __post_init__(self):
        self.data_dir = self.data_dir.expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.credit.db_path = self.data_dir / "credits.db"
