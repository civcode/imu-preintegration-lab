"""Record FT232H + MPU-6050 FIFO samples to an SI-unit CSV and metadata JSON."""
import argparse
from pathlib import Path

from imu_lab.calibration import Calibration, calibration_callback
from imu_lab.ftdi_mpu6050 import MPU6050Ftdi
from imu_lab.recording import save_capture


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/stationary.csv"))
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--url", default="ftdi://ftdi:232h/1")
    parser.add_argument("--sample-div", type=int, default=4, help="1kHz/(1+div), default 200Hz")
    parser.add_argument("--calibration", type=Path, help="Validated JSON calibration, optional")
    args = parser.parse_args(argv)
    params = Calibration.from_json(args.calibration) if args.calibration else None
    print("Place the sensor on a stable surface for the full stationary test.")
    result = None
    try:
        with MPU6050Ftdi(url=args.url, sample_div=args.sample_div,
                         calibration=calibration_callback if params else None,
                         calibration_data=params) as imu:
            result = imu.capture(duration_s=args.seconds)
    except KeyboardInterrupt:
        # Handles Ctrl+C during setup/teardown, not only during capture.
        print("\nInterrupted while opening/closing FTDI. Interface cleanup attempted.")
        return 130
    if result is None:
        print("No capture result; nothing to save.")
        return 1
    # Even if overflow occurred before the first complete packet, save its
    # diagnostic metadata rather than suppressing an important sensor error.
    if result.overflow_detected or result.samples:
        save_capture(result, args.out, calibrated=params is not None)
    else:
        print("No samples received; nothing to save.")
        return 1
    print(f"Saved {len(result.samples)} samples in {args.out}; status={result.stop_reason}")
    if result.overflow_detected:
        print("WARNING: FIFO OVERFLOW. This recording is not usable as one contiguous IMU path.")
        return 2
    return 0 if result.stop_reason != "interrupted" else 130


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted safely (Ctrl+C).")
        raise SystemExit(130)
