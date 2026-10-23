"""CSV I/O for FIFO capture. Only calibrated measurements are converted to SI."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
from .core import GRAVITY
from .ftdi_mpu6050 import CaptureResult


def save_capture(result: CaptureResult, path: str | Path, *, calibrated: bool) -> None:
    """Save data and sidecar metadata. Overflow recordings are not valid motion tracks.

    Always save even an interrupted partial recording when at least one sample
    exists, so Ctrl+C never silently discards the student's measurements.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t_s", "ax", "ay", "az", "gx", "gy", "gz", "temp_c"])
        for s in result.samples:
            a = np.asarray(s["a"], float) * GRAVITY
            w = np.deg2rad(np.asarray(s["w"], float))
            writer.writerow([f"{s['t']:.9f}", *a, *w, s.get("temp", "")])
    info = {
        "source": "FT232H / MPU-6050 FIFO", "timebase": "nominal sample-index / sample-rate",
        "sample_rate_hz": result.sample_rate_hz, "sample_count": len(result.samples),
        "stop_reason": result.stop_reason, "overflow_detected": result.overflow_detected,
        "peak_fifo_bytes": result.peak_fifo_bytes, "elapsed_host_s": result.elapsed_host_s,
        "started_utc": result.started_utc, "calibrated": calibrated,
        "accelerometer_units": "m/s^2 specific force", "gyroscope_units": "rad/s",
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf8")


def verify_capture_metadata(path: str | Path) -> None:
    """Reject known-broken sequences rather than integrating over lost packets."""
    path = Path(path)
    meta = path.with_suffix(".metadata.json")
    if not meta.exists():
        return  # Synthetic data and third-party CSVs need not have metadata.
    data = json.loads(meta.read_text(encoding="utf8"))
    if data.get("overflow_detected", False):
        raise ValueError(f"FIFO overflow recorded in {meta}. Do not integrate across the gap.")
