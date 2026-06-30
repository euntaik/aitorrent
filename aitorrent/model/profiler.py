from __future__ import annotations

import time
from dataclasses import dataclass

import psutil
import torch


@dataclass
class HardwareProfile:
    gpu_name: str | None
    gpu_vram_total_mb: int
    gpu_vram_free_mb: int
    ram_total_mb: int
    ram_free_mb: int
    cpu_cores: int
    compute_tflops: float

    def max_layers(self, params_per_layer: int, dtype_bytes: int = 2) -> int:
        vram = self.gpu_vram_free_mb if self.gpu_name else 0
        usable_mb = max(vram, self.ram_free_mb) * 0.85
        layer_mb = (params_per_layer * dtype_bytes) / (1024 * 1024)
        if layer_mb == 0:
            return 0
        return int(usable_mb / layer_mb)


class HardwareProfiler:
    def profile(self) -> HardwareProfile:
        mem = psutil.virtual_memory()
        gpu_name, vram_total, vram_free = self._detect_gpu()
        tflops = self._benchmark_compute()
        return HardwareProfile(
            gpu_name=gpu_name,
            gpu_vram_total_mb=vram_total,
            gpu_vram_free_mb=vram_free,
            ram_total_mb=int(mem.total / (1024 * 1024)),
            ram_free_mb=int(mem.available / (1024 * 1024)),
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            compute_tflops=tflops,
        )

    def _detect_gpu(self) -> tuple[str | None, int, int]:
        if not torch.cuda.is_available():
            return None, 0, 0
        try:
            props = torch.cuda.get_device_properties(0)
            name = props.name
            total = int(props.total_mem / (1024 * 1024))
            free = total - int(torch.cuda.memory_allocated(0) / (1024 * 1024))
            return name, total, free
        except Exception:
            return None, 0, 0

    def _benchmark_compute(
        self, matrix_size: int = 2048, iterations: int = 50
    ) -> float:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)

        # warmup
        for _ in range(5):
            torch.mm(a, b)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            torch.mm(a, b)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        flops_per_op = 2 * matrix_size**3
        total_flops = flops_per_op * iterations
        tflops = total_flops / elapsed / 1e12
        return round(tflops, 2)
