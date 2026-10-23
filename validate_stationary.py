"""Diagnose a motionless MPU-6050 recording before trying 3D motion.

Run on a stationary (~20 s) recording. The output tells you whether sample
spacing and gravity are plausible, and makes uncompensated drift visible.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from imu_lab import GRAVITY, estimate_initial_state, integrate_direct, load_csv
from imu_lab.recording import verify_capture_metadata


def diagnose(path: str | Path) -> dict:
    verify_capture_metadata(path)
    imu = load_csv(path)
    dt = np.diff(imu.t)
    R0, residual_gyro_bias = estimate_initial_state(
        imu, static_seconds=min(2., float(imu.t[-1] - imu.t[0]) * 0.5))
    raw = integrate_direct(imu, R0=R0)
    corrected = integrate_direct(imu, R0=R0, gyro_bias=residual_gyro_bias)
    info = {
        "duration_s": imu.t[-1] - imu.t[0],
        "samples": len(imu.t),
        "dt_mean_ms": 1000 * dt.mean(),
        "dt_std_ms": 1000 * dt.std(),
        "accel_mean_norm_m_s2": np.linalg.norm(imu.acc.mean(axis=0)),
        "accel_std_m_s2": imu.acc.std(axis=0),
        "gyro_mean_deg_s": np.rad2deg(imu.gyro.mean(axis=0)),
        "gyro_std_deg_s": np.rad2deg(imu.gyro.std(axis=0)),
        "uncorrected_endpoint_m": np.linalg.norm(raw.p[-1]),
        "initial_bias_removed_endpoint_m": np.linalg.norm(corrected.p[-1]),
        "trajectory_raw": raw,
        "trajectory_corrected": corrected,
    }
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", type=Path, help="Stationary recording in SI-unit CSV")
    ap.add_argument("--plot", type=Path, help="Save XY path comparison to PNG")
    args = ap.parse_args(argv)
    out = diagnose(args.file)
    print(f"Duration: {out['duration_s']:.2f} s, samples: {out['samples']}")
    print(f"dt: {out['dt_mean_ms']:.4f} ± {out['dt_std_ms']:.4f} ms (nominal FIFO timebase)")
    print(f"Mean |accelerometer|: {out['accel_mean_norm_m_s2']:.5f} m/s² (expected about {GRAVITY:g})")
    print(f"Accelerometer std [m/s²]: {out['accel_std_m_s2']}")
    print(f"Gyro mean [degree/s]: {out['gyro_mean_deg_s']}")
    print(f"Gyro std [degree/s]: {out['gyro_std_deg_s']}")
    print(f"Raw endpoint error at rest: {out['uncorrected_endpoint_m']:.4f} m")
    print(f"With initial gyro bias removed: {out['initial_bias_removed_endpoint_m']:.4f} m")
    print("Note: these are apparent drift under a known zero-motion condition, not position accuracy.")
    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for key, label in (("trajectory_raw", "raw gyro"),
                           ("trajectory_corrected", "first-2s gyro bias removed")):
            path = out[key].p
            ax.plot(path[:, 0], path[:, 1], label=label)
        ax.scatter([0], [0], c="black", marker="x", label="true stationary position")
        ax.set(xlabel="X (m)", ylabel="Y (m)", title="IMU-only drift while stationary")
        ax.axis("equal")
        ax.legend()
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot)
        plt.close(fig)
        print(f"Plot: {args.plot}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C. Nothing left running.")
        raise SystemExit(130)
