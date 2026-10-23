"""Physical, sequential cube-based calibration of an MPU-6050.

The stages are NOT interchangeable. Measurements are raw and retained;
corrections are applied in order when selecting and fitting subsequent stages:

    multi-orientation accel -> six-face cube alignment -> zero-rate gyro
        -> six signed rotation experiments -> independent validation

This module deliberately leaves thresholds with the experimenter and reports
sample counts/residuals. It never invents sensor calibration from demo data.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .calibration import (
    Calibration, estimate_gyro_bias, estimate_six_pose_rotation,
    fit_accelerometer_ellipsoid, fit_gyro_axis_matrix,
)
from .ftdi_mpu6050 import CaptureResult, MPU6050Ftdi

FACES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
AXES = {"+X": np.array([1., 0., 0.]), "-X": np.array([-1., 0., 0.]),
        "+Y": np.array([0., 1., 0.]), "-Y": np.array([0., -1., 0.]),
        "+Z": np.array([0., 0., 1.]), "-Z": np.array([0., 0., -1.])}
STAGES = ("01_ellipsoid", "02_cube_alignment", "03_gyro_bias", "04_gyro_axes")
PREVIOUS = dict(zip(STAGES[1:], STAGES[:-1]))


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf8")


def record_raw_stage(
    out_csv: str | Path, *, seconds: float, url: str = "ftdi://ftdi:232h/1",
    bus_rate: int = 400_000, sample_div: int = 4, dlpf_cfg: int = 3,
    overwrite: bool = False, verbose: bool = True,
    controller_factory=None,
) -> CaptureResult:
    """Capture *uncalibrated* g / degree/s without losing samples on Ctrl+C.

    The context manager stops FIFO and closes FTDI even on interruptions.
    CaptureResult stops with reason 'interrupted' on Ctrl+C. All complete
    samples already collected are written immediately, even if interrupted.
    Bad/overflow recordings are saved for diagnosis, never accepted as valid.
    Neither this method nor the FIFO can recover missing hardware timestamps.
    """
    path = Path(out_csv)
    if not re.fullmatch(r"[a-zA-Z0-9_+.-]+", path.stem):
        raise ValueError("Use an ordinary recording filename")
    meta = path.with_suffix(".metadata.json")
    if not overwrite and (path.exists() or meta.exists()):
        raise FileExistsError(f"{path} already exists; reuse it or explicitly enable overwrite")
    with MPU6050Ftdi(url=url, bus_rate=bus_rate, sample_div=sample_div,
                     dlpf_cfg=dlpf_cfg, verbose=verbose,
                     controller_factory=controller_factory) as imu:
        result = imu.capture(duration_s=seconds, include_temp=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("t_s", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"))
        for s in result.samples:
            writer.writerow((f"{s['t']:.9f}", *s["a"], *s["w"]))
    _json(meta, {
        "source": "MPU-6050 FT232H FIFO (raw, before calibration)",
        "accelerometer_units": "g", "gyroscope_units": "degree/s",
        "sensor_range_accel_g": 2, "sensor_range_gyro_dps": 250,
        "sample_div": sample_div, "dlpf_cfg": dlpf_cfg,
        "sample_rate_hz": result.sample_rate_hz,
        "stop_reason": result.stop_reason, "overflow_detected": result.overflow_detected,
        "samples": len(result.samples), "peak_fifo_bytes": result.peak_fifo_bytes,
        "elapsed_host_s": result.elapsed_host_s, "started_utc": result.started_utc,
        "nominal_clock_warning": "No individual hardware timestamps. Overflow invalidates continuity."
    })
    return result


def load_raw_stage(path: str | Path, *, allow_interrupted: bool = False,
                   sample_div: int = 4, dlpf_cfg: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one unchanged raw recording; reject overflow or settings mismatch."""
    path = Path(path)
    info = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf8"))
    if (info.get("sample_div"), info.get("dlpf_cfg")) != (sample_div, dlpf_cfg):
        raise ValueError(f"Sensor settings differ in {path}; cannot combine calibration datasets")
    if info.get("overflow_detected") or info.get("stop_reason") == "fifo_overflow":
        raise ValueError(f"FIFO overflow in {path}; repeat this recording")
    if info.get("stop_reason") == "interrupted" and not allow_interrupted:
        raise ValueError(f"{path} ended with Ctrl+C: set allow_interrupted=True only after inspection")
    if info.get("stop_reason") not in ("completed", "max_samples", "interrupted"):
        raise ValueError(f"Unusable recording: {info.get('stop_reason')}")
    with path.open(newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) != info.get("samples"):
        raise ValueError(f"Empty or truncated recording: {path}")
    data = np.asarray([[float(r[k]) for k in ("t_s", "ax_g", "ay_g", "az_g",
                                                "gx_dps", "gy_dps", "gz_dps")] for r in rows])
    if not np.isfinite(data).all() or (len(data) > 1 and not np.allclose(
            np.diff(data[:, 0]), 1 / info["sample_rate_hz"], atol=1e-6, rtol=0)):
        raise ValueError(f"Invalid sensor timestamps or samples in {path}")
    return data[:, 0], data[:, 1:4], data[:, 4:7]


def still_mask(gyro_dps: np.ndarray, max_rate_dps: float, *, bias_dps=None) -> np.ndarray:
    """Select stationary samples; inspect rate histogram before picking threshold.

    Before measuring gyro bias, pass None to reproduce the source notebook's
    absolute-rate threshold (its starting example was 3.7 degree/s).
    Afterwards subtract the measured bias for more meaningful screening.
    """
    if max_rate_dps <= 0:
        raise ValueError("Stationary gyro threshold must be positive")
    w = np.asarray(gyro_dps, float)
    return np.linalg.norm(w - (np.zeros(3) if bias_dps is None else bias_dps), axis=1) < max_rate_dps


def fit_cube_faces(raw_data: dict[str, tuple[np.ndarray, np.ndarray]],
                   bias: np.ndarray, L: np.ndarray, max_rate_dps: float,
                   min_samples: int = 30) -> tuple[np.ndarray, dict]:
    """Each named cube pose measures the corresponding +/− gravity direction.

    Data are recorded RAW; apply the stage-1 ellipsoid parameters before this
    alignment, rather than selecting faces in uncorrected sensor axes.
    """
    means, counts = [], {}
    for face in FACES:
        if face not in raw_data:
            raise ValueError(f"Missing physical cube placement {face}")
        a, w = raw_data[face]
        good = still_mask(w, max_rate_dps)
        counts[face] = int(good.sum())
        if good.sum() < min_samples:
            raise ValueError(f"{face}: only {good.sum()} still samples, need {min_samples}")
        means.append(np.mean((a[good] - bias) @ L.T, axis=0))
    means = np.asarray(means)
    ref = np.stack([AXES[f] for f in FACES])
    R = estimate_six_pose_rotation(means, ref)
    calibrated = (R @ means.T).T
    angles = np.rad2deg(np.arccos(np.clip(
        np.sum(calibrated * ref, axis=1) / np.linalg.norm(calibrated, axis=1), -1, 1)))
    return R, {"counts": counts, "angular_residual_deg": dict(zip(FACES, angles.tolist())),
               "gravity_norm_g": dict(zip(FACES, np.linalg.norm(means, axis=1).tolist()))}


def fit_stationary_gyro(a: np.ndarray, w: np.ndarray, previous: Calibration,
                        *, max_gravity_error_g: float = 0.15,
                        max_gyro_std_dps: float = 1.0) -> tuple[np.ndarray, dict]:
    """Gyro bias from stationary data after verifying corrected gravity norm.

    Ellipsoid AND six-face calibration are now available for cross-checking.
    No gyro bias is subtracted while estimating the bias itself.
    """
    ac = (previous.accel_body_rotation @ (previous.accel_matrix @
          (np.asarray(a, float) - previous.accel_bias_g).T)).T
    err = np.abs(np.linalg.norm(ac, axis=1) - 1.0)
    mask = err < max_gravity_error_g
    if mask.sum() < 30:
        raise ValueError("Too few near-1g samples; set cube down and record again")
    std = np.std(np.asarray(w)[mask], axis=0)
    if max_gyro_std_dps <= 0 or np.any(std > max_gyro_std_dps):
        raise ValueError(f"Gyro fluctuates too much for stationary bias: {std} degree/s")
    return estimate_gyro_bias(w[mask]), {"selected": int(mask.sum()),
        "rejected": int((~mask).sum()), "gravity_error_rms_g": float(np.sqrt(np.mean(err[mask]**2))),
        "gyro_std_dps": np.std(np.asarray(w)[mask], axis=0).tolist()}


def rotation_mask(w: np.ndarray, face: str, *, min_dps: float, max_dps: float,
                  direction_cosine: float, max_subordinate_dps: float) -> np.ndarray:
    """Select samples for a *labeled physical rotation* in gyro raw axes.

    These are the original notebook's four separate gates, now configurable
    based on the observed 6-trial data and sensor full scale (±250 deg/s).
    Beware: too-small subordinate limit can conceal real gyro-axis crosstalk.
    """
    if face not in FACES:
        raise ValueError(f"Unknown signed body rotation {face}")
    if not (0 <= min_dps < max_dps < 250):
        raise ValueError("Need 0 <= min < max < ±250 dps sensor full scale")
    if not 0 < direction_cosine <= 1 or max_subordinate_dps <= 0:
        raise ValueError("Invalid direction cosine or subordinate-axis limit")
    w = np.asarray(w, float)
    idx = FACES.index(face) // 2
    axis = AXES[face]
    mag = np.linalg.norm(w, axis=1)
    directed = w @ axis
    subordinate = np.linalg.norm(np.delete(w, idx, axis=1), axis=1)
    return (directed >= min_dps) & (directed <= max_dps) & (
        directed / np.maximum(mag, 1e-12) >= direction_cosine) & (
        subordinate <= max_subordinate_dps)


def fit_signed_gyro_rotations(rates: dict[str, np.ndarray], *,
                              min_dps: float, max_dps: float,
                              direction_cosine: float = .9,
                              max_subordinate_dps: float = 10.,
                              min_samples: int = 20) -> tuple[np.ndarray, dict]:
    """Use + and − cube-axis turns to estimate normalized sensor gyro axes.

    INPUT is already corrected for STAGE-3 stationary bias. Reverse negative
    samples, combine with positive ones, estimate axes then invert U.
    Rotating by known axes identifies axis directions, NOT absolute gains.
    """
    centers, counts, selections, opposite_angles = [], {}, {}, {}
    for f in FACES:
        if f not in rates:
            raise ValueError(f"Missing signed physical rotation {f}")
        w = np.asarray(rates[f], float)
        mask = rotation_mask(w, f, min_dps=min_dps, max_dps=max_dps,
                            direction_cosine=direction_cosine,
                            max_subordinate_dps=max_subordinate_dps)
        counts[f] = int(mask.sum())
        selections[f] = w[mask]
        if mask.sum() < min_samples:
            raise ValueError(f"{f}: {mask.sum()} selected rotations; tune gates or recapture (need {min_samples})")
    for pos, neg in zip(FACES[::2], FACES[1::2]):
        both = np.vstack([selections[pos], -selections[neg]])
        positive = selections[pos].mean(axis=0)
        negative_reversed = -selections[neg].mean(axis=0)
        opposite_angles[f"{pos}/{neg}"] = float(np.rad2deg(np.arccos(np.clip(
            positive @ negative_reversed / (
                np.linalg.norm(positive) * np.linalg.norm(negative_reversed)), -1, 1))))
        centers.append(both.mean(axis=0))
    centers = np.asarray(centers)
    T = fit_gyro_axis_matrix(centers)
    U = np.linalg.inv(T)
    condition = float(np.linalg.cond(U))
    if not np.isfinite(condition) or condition > 20:
        raise ValueError(f"Badly conditioned gyro-axis matrix ({condition:.1f}); recapture")
    return T, {"selected_counts": counts, "mean_rate_vectors_dps": centers.tolist(),
               "condition_number": condition,
               "positive_negative_angle_deg": opposite_angles}


class StageStore:
    """Dependency-aware JSON checkpoints prevent silently using stale stages.

    Every checkpoint fingerprints raw capture files and its predecessor.
    If you change a recording, threshold or upstream result and resave,
    the downstream checkpoints become invalid until recomputed.
    """
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def file(self, stage: str) -> Path:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage}")
        return self.directory / f"{stage}.json"

    def load(self, stage: str) -> dict:
        path = self.file(stage)
        record = json.loads(path.read_text(encoding="utf8"))
        if record.get("stage") != stage:
            raise ValueError(f"Checkpoint stage mismatch: {path}")
        parent = PREVIOUS.get(stage)
        if parent:
            self.load(parent)  # validate the entire chain, not just one step
            if record.get("parent_sha256") != _sha(self.file(parent)):
                raise ValueError(f"{stage} is stale: previous calibration changed; refit this stage")
        for filepath, digest in record["sources"].items():
            if not Path(filepath).exists() or _sha(Path(filepath)) != digest:
                raise ValueError(f"{stage} is stale: original data changed or disappeared: {filepath}")
        return record["payload"]

    def save(self, stage: str, payload: dict, *, raw_files: list[str | Path],
             settings: dict, overwrite: bool = False) -> Path:
        path = self.file(stage)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists; use overwrite=True when refitting")
        parent = PREVIOUS.get(stage)
        if parent:
            self.load(parent)
        sources = {}
        for raw in raw_files:
            raw_path = Path(raw).resolve()
            meta = raw_path.with_suffix(".metadata.json")
            for file in (raw_path, meta):
                sources[str(file)] = _sha(file)
        obj = {"stage": stage, "created_utc": datetime.now(timezone.utc).isoformat(),
               "parent_sha256": _sha(self.file(parent)) if parent else None,
               "sources": sources, "selection_settings": settings, "payload": payload}
        _json(path, obj)
        return path

    def final_calibration(self) -> Calibration:
        """Only export after all four current stage checkpoints validate."""
        self.load(STAGES[-1])
        e, a, g, t = (self.load(stage) for stage in STAGES)
        return Calibration(np.asarray(e["bias_g"]), np.asarray(e["L"]),
                           np.asarray(a["R"]), np.asarray(g["bias_dps"]),
                           np.asarray(t["T"])).validate()
