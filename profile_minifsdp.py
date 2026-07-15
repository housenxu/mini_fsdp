from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    trace_dir = Path("traces")
    cmd = [
        sys.executable,
        "benchmark.py",
        "--model",
        "mlp",
        "--strategy",
        "minifsdp-layerwise",
        "--steps",
        "10",
        "--warmup",
        "2",
        "--profile",
        "--trace-dir",
        str(trace_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Profiler traces written to: {trace_dir.resolve()}")
    print("Open with: tensorboard --logdir traces")


if __name__ == "__main__":
    main()
