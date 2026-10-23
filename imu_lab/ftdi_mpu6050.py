"""MPU-6050 FIFO recorder using the FT232H's I²C interface (PyFtdi).

Teaching point: FIFO samples already exist before USB reads them. They do NOT
occur when Python unpacks their bytes. Give each sample a nominal sensor time
(index / sample_rate_hz), and explicitly fail/flag on any possible data loss.

This is NOT a hardware-clock synchronization solution: MPU-6050 sample clock
error can produce drift in the nominal time scale. A FIFO has no per-sample
hardware timestamp and cannot recover the number of samples lost on overflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import struct
import time
from typing import Any, Callable

CalibrationCallback = Callable[[dict[str, Any], Any], dict[str, Any]]


@dataclass
class CaptureResult:
    samples: list[dict[str, Any]]
    sample_rate_hz: float
    stop_reason: str
    peak_fifo_bytes: int
    elapsed_host_s: float
    started_utc: str
    overflow_detected: bool = False


class MPU6050Ftdi:
    """Acquire calibrated acceleration (g) and angular rate (degrees/s).

    Always use ``with MPU6050Ftdi(...) as imu:`` for USB/I²C cleanup. A
    supplied ``controller_factory`` allows the entire capture logic to be
    tested with a fake I²C device, without connecting hardware.
    """

    ADDRESS = 0x68
    REG_SMPLRT_DIV = 0x19
    REG_CONFIG = 0x1A
    REG_GYRO_CONFIG = 0x1B
    REG_ACCEL_CONFIG = 0x1C
    REG_FIFO_EN = 0x23
    REG_INT_ENABLE = 0x38
    REG_INT_STATUS = 0x3A
    REG_TEMP_OUT_H = 0x41
    REG_USER_CTRL = 0x6A
    REG_PWR_MGMT_1 = 0x6B
    REG_PWR_MGMT_2 = 0x6C
    REG_FIFO_COUNTH = 0x72
    REG_FIFO_R_W = 0x74
    REG_WHO_AM_I = 0x75

    BYTES_PER_SAMPLE = 12  # three signed 16-bit accel + three signed 16-bit gyro
    FIFO_CAPACITY = 1024
    INT_FIFO_OVERFLOW = 0x10
    ACCEL_SCALE = 16384.0  # LSB/g, only for +/-2g
    GYRO_SCALE = 131.0  # LSB/(degree/s), only for +/-250 degree/s

    def __init__(
        self,
        url: str = "ftdi://ftdi:232h/1",
        bus_rate: int = 400_000,
        addr: int = ADDRESS,
        sample_div: int = 4,
        dlpf_cfg: int = 3,
        calibration: CalibrationCallback | None = None,
        calibration_data: Any = None,
        verbose: bool = True,
        controller_factory: Callable[[], Any] | None = None,
    ) -> None:
        # With DLPF_CFG 1..6 the underlying sample generator is 1 kHz.
        # At DLPF_CFG 0 or 7 the gyro generator is 8 kHz; do not silently
        # calculate the wrong time base for a teaching experiment.
        if not (1 <= dlpf_cfg <= 6):
            raise ValueError("This lab requires DLPF_CFG in 1..6 (1 kHz base rate).")
        if not (0 <= sample_div <= 255):
            raise ValueError("sample_div must be between 0 and 255")
        self.sample_rate_hz = 1000.0 / (1 + sample_div)
        self.verbose = verbose
        self.calibration = calibration
        self.calibration_data = calibration_data
        self._closed = False
        self._recording = False
        self.i2c = None
        try:
            if controller_factory is None:
                from pyftdi.i2c import I2cController  # optional until hardware use
                controller_factory = I2cController
            self.i2c = controller_factory()
            self.i2c.configure(f"{url}?frequency={bus_rate}")
            self.port = self.i2c.get_port(addr)
            who = self._read8(self.REG_WHO_AM_I)
            if who != 0x68:
                raise RuntimeError(f"Unexpected WHO_AM_I 0x{who:02X}; expected 0x68")
            self._configure(sample_div, dlpf_cfg)
        except BaseException:
            # Includes KeyboardInterrupt during initialization.
            self.close()
            raise

    def _write8(self, reg: int, value: int) -> None:
        self.port.write_to(reg, bytes([value & 0xFF]))

    def _read8(self, reg: int) -> int:
        return self.port.exchange(bytes([reg]), 1)[0]

    def _read16(self, reg: int) -> int:
        return struct.unpack(">H", self.port.exchange(bytes([reg]), 2))[0]

    def _configure(self, sample_div: int, dlpf_cfg: int) -> None:
        self._write8(self.REG_PWR_MGMT_1, 0x01)  # X gyro PLL clock
        self._write8(self.REG_PWR_MGMT_2, 0x00)  # enable all sensor axes
        self._write8(self.REG_CONFIG, dlpf_cfg)
        self._write8(self.REG_SMPLRT_DIV, sample_div)
        self._write8(self.REG_GYRO_CONFIG, 0x00)  # +/-250 degrees/s
        self._write8(self.REG_ACCEL_CONFIG, 0x00)  # +/-2g
        self._stop_fifo()

    def _stop_fifo(self) -> None:
        self._write8(self.REG_FIFO_EN, 0x00)  # stop adding samples
        self._write8(self.REG_USER_CTRL, 0x00)  # FIFO logic off
        self._recording = False

    def _start_fifo(self) -> None:
        # Crucial: initialize the timing epoch AFTER setup/diagnostic delays.
        # Otherwise old samples waiting in FIFO would be mistaken for new data.
        self._stop_fifo()
        self._write8(self.REG_USER_CTRL, 0x04)  # FIFO_RESET pulse
        self._write8(self.REG_USER_CTRL, 0x40)  # FIFO_EN logic
        self._write8(self.REG_FIFO_EN, 0x78)  # accel + gyro XYZ, no temperature
        self._write8(self.REG_INT_ENABLE, self.INT_FIFO_OVERFLOW)
        self._read8(self.REG_INT_STATUS)  # clear old latched status
        self._recording = True

    def capture(
        self,
        duration_s: float | None = 20.0,
        max_samples: int | None = None,
        poll_sleep_s: float = 0.002,
        include_temp: bool = True,
    ) -> CaptureResult:
        """Collect FIFO data, stopping cleanly on Ctrl+C or on overflow.

        ``duration_s`` refers to nominal sensor time, not the USB receive time.
        Full packets always get t=sample_index/sample_rate_hz. One clock period
        corresponds to 5 ms at the default 200 Hz. On overflow we STOP and
        mark the recording invalid for continuous integration; never pretend
        a FIFO reset preserved elapsed sample counts.
        """
        if duration_s is None and max_samples is None:
            raise ValueError("Specify duration_s or max_samples to bound recording")
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive")
        samples: list[dict[str, Any]] = []
        started = datetime.now(timezone.utc).isoformat()
        stop_reason = "completed"
        peak_fifo_bytes = 0
        t0 = time.monotonic()
        self._start_fifo()
        try:
            while True:
                if max_samples is not None and len(samples) >= max_samples:
                    stop_reason = "max_samples"
                    break
                if duration_s is not None and len(samples) >= round(duration_s * self.sample_rate_hz):
                    break

                # Status is latched. FIFO_COUNT near its limit is additionally
                # dangerous even if the interrupt has not been observed yet.
                status = self._read8(self.REG_INT_STATUS)
                fifo_count = self._read16(self.REG_FIFO_COUNTH)
                peak_fifo_bytes = max(peak_fifo_bytes, fifo_count)
                if (status & self.INT_FIFO_OVERFLOW) or fifo_count >= self.FIFO_CAPACITY:
                    stop_reason = "fifo_overflow"
                    break
                n_packets = fifo_count // self.BYTES_PER_SAMPLE
                if n_packets == 0:
                    time.sleep(poll_sleep_s)
                    continue
                remaining = float('inf') if max_samples is None else max_samples - len(samples)
                duration_remaining = (float('inf') if duration_s is None
                                      else round(duration_s * self.sample_rate_hz) - len(samples))
                # Bounded I²C transfers allow other Python interrupts to be
                # serviced, and avoid large FTDI transaction buffers.
                n_packets = min(n_packets, 20, remaining, duration_remaining)
                blob = self.port.exchange(bytes([self.REG_FIFO_R_W]), int(n_packets) * self.BYTES_PER_SAMPLE)
                if len(blob) != int(n_packets) * self.BYTES_PER_SAMPLE:
                    raise IOError("Short FIFO read; cannot trust IMU sample continuity")
                temperature = self.read_temperature() if include_temp else None
                for packet in struct.iter_unpack(">hhhhhh", blob):
                    ax, ay, az, gx, gy, gz = packet
                    record = {
                        "t": len(samples) / self.sample_rate_hz,
                        "a": [ax / self.ACCEL_SCALE, ay / self.ACCEL_SCALE, az / self.ACCEL_SCALE],
                        "w": [gx / self.GYRO_SCALE, gy / self.GYRO_SCALE, gz / self.GYRO_SCALE],
                    }
                    if self.calibration is not None:
                        record = self.calibration(record, self.calibration_data)
                    if temperature is not None:
                        record["temp"] = temperature
                    samples.append(record)
        except KeyboardInterrupt:
            # Jupyter's Interrupt Kernel and terminal Ctrl+C normally raise
            # KeyboardInterrupt. Return collected measurements, not a traceback.
            stop_reason = "interrupted"
            if self.verbose:
                print("\nCapture interrupted: safely stopping FIFO and returning collected samples.")
        finally:
            self._stop_fifo()  # context manager also closes USB on exit
        elapsed = time.monotonic() - t0
        if self.verbose:
            print(f"Capture: {len(samples)} samples at nominal {self.sample_rate_hz:g} Hz; "
                  f"reason={stop_reason}; peak FIFO={peak_fifo_bytes} bytes; "
                  f"host elapsed={elapsed:.2f}s")
        return CaptureResult(
            samples=samples, sample_rate_hz=self.sample_rate_hz,
            stop_reason=stop_reason, peak_fifo_bytes=peak_fifo_bytes,
            elapsed_host_s=elapsed, started_utc=started,
            overflow_detected=(stop_reason == "fifo_overflow"),
        )

    def read_temperature(self) -> float:
        raw = struct.unpack(">h", self.port.exchange(bytes([self.REG_TEMP_OUT_H]), 2))[0]
        return raw / 340.0 + 36.53

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.i2c is None:
            return
        try:
            if getattr(self, "port", None) is not None:
                self._stop_fifo()
        except Exception:
            pass  # disconnects can prevent register writes
        finally:
            self.i2c.terminate()

    def __enter__(self) -> "MPU6050Ftdi":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
