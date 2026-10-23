"""No real USB device needed: emulate the register and FIFO protocol."""
import struct
import numpy as np
import pytest
from imu_lab.calibration import Calibration, fit_accelerometer_ellipsoid, estimate_gyro_bias
from imu_lab.ftdi_mpu6050 import MPU6050Ftdi
from imu_lab.recording import save_capture, verify_capture_metadata


class FakePort:
    def __init__(self, packet_count=5, interrupt_after=None, overflow=False):
        self.packet_count = packet_count
        self.interrupt_after = interrupt_after
        self.overflow = overflow
        self.read_packets = 0
        self.enabled = False
        self.buffer = b''
        self.write_calls = []

    def write_to(self, reg, data):
        self.write_calls.append((reg, bytes(data)))
        if reg == MPU6050Ftdi.REG_FIFO_EN:
            self.enabled = bool(data[0])
        if reg == MPU6050Ftdi.REG_USER_CTRL and data[0] & 0x04:
            self.buffer = b''

    def exchange(self, address, count):
        reg = address[0]
        if reg == MPU6050Ftdi.REG_WHO_AM_I:
            return b'\x68'
        if reg == MPU6050Ftdi.REG_INT_STATUS:
            return b'\x10' if self.enabled and self.overflow else b'\x00'
        if reg == MPU6050Ftdi.REG_FIFO_COUNTH:
            if self.interrupt_after is not None and self.read_packets >= self.interrupt_after:
                raise KeyboardInterrupt
            if self.enabled and not self.buffer and self.read_packets < self.packet_count:
                for _ in range(min(4, self.packet_count - self.read_packets)):
                    self.buffer += struct.pack('>hhhhhh', 0, 0, 16384, 0, 0, 131)
            return struct.pack('>H', len(self.buffer))
        if reg == MPU6050Ftdi.REG_FIFO_R_W:
            blob, self.buffer = self.buffer[:count], self.buffer[count:]
            self.read_packets += count // 12
            return blob
        if reg == MPU6050Ftdi.REG_TEMP_OUT_H:
            return struct.pack('>h', 0)
        raise AssertionError(f'unhandled register {reg}')


class FakeController:
    def __init__(self, port):
        self.port = port
        self.terminated = False

    def configure(self, url):
        self.url = url

    def get_port(self, addr):
        return self.port

    def terminate(self):
        self.terminated = True


def make_device(fake):
    controller = FakeController(fake)
    device = MPU6050Ftdi(verbose=False, controller_factory=lambda: controller)
    return controller, device


def test_fifo_sensor_timestamps_and_cleanup(tmp_path):
    controller, device = make_device(FakePort(packet_count=7))
    with device as sensor:
        result = sensor.capture(max_samples=7, duration_s=None, include_temp=False)
    assert controller.terminated
    assert result.stop_reason == 'max_samples'
    np.testing.assert_allclose([s['t'] for s in result.samples], np.arange(7) / 200)
    np.testing.assert_allclose([s['a'] for s in result.samples], np.tile([0, 0, 1], (7, 1)))
    assert all(s['w'] == [0, 0, 1] for s in result.samples)
    save_capture(result, tmp_path / 'capture.csv', calibrated=False)
    verify_capture_metadata(tmp_path / 'capture.csv')


def test_ctrl_c_preserves_partial_samples_and_closes_port():
    controller, device = make_device(FakePort(packet_count=10, interrupt_after=4))
    with device as sensor:
        result = sensor.capture(max_samples=10, duration_s=None, include_temp=False)
    assert result.stop_reason == 'interrupted'
    assert len(result.samples) == 4
    assert controller.terminated


def test_overflow_is_reported_and_bad_capture_rejected(tmp_path):
    controller, device = make_device(FakePort(overflow=True))
    with device as sensor:
        result = sensor.capture(max_samples=10, duration_s=None, include_temp=False)
    assert result.overflow_detected and result.stop_reason == 'fifo_overflow'
    assert controller.terminated
    save_capture(result, tmp_path / 'capture.csv', calibrated=False)
    with pytest.raises(ValueError, match='FIFO overflow'):
        verify_capture_metadata(tmp_path / 'capture.csv')


def test_error_does_not_get_hidden_in_finally():
    fake = FakePort(packet_count=2)
    controller, device = make_device(fake)
    fake.exchange = lambda *_: (_ for _ in ()).throw(OSError('USB disconnected'))
    with pytest.raises(OSError, match='USB disconnected'):
        with device as sensor:
            sensor.capture(max_samples=2, duration_s=None)
    assert controller.terminated


def test_calibration_units_and_json(tmp_path):
    c = Calibration.identity()
    c.accel_bias_g = np.array([.1, 0, 0])
    c.gyro_bias_dps = np.array([0, 0, 1.])
    p = tmp_path / 'calibration.json'
    c.to_json(p)
    c = Calibration.from_json(p)
    raw = {'a': [.1, 0, 1.], 'w': [0, 0, 1.]}
    corrected = c.apply(raw)
    np.testing.assert_allclose(corrected['a'], [0, 0, 1])
    np.testing.assert_allclose(corrected['w'], [0, 0, 0])
    assert raw['a'][0] == .1  # callback must not mutate the original sample


def test_accel_fit_on_known_static_orientations():
    rng = np.random.default_rng(7)
    v = rng.normal(size=(500, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    b = np.array([.05, -.03, .02])
    measured = b + v @ np.linalg.inv(np.diag([1.02, .97, 1.03])).T
    est_bias, L = fit_accelerometer_ellipsoid(measured)
    np.testing.assert_allclose(est_bias, b, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm((measured - est_bias) @ L.T, axis=1), 1., atol=1e-3)
    assert estimate_gyro_bias(np.ones((20, 3))).tolist() == [1, 1, 1]


def test_convert_legacy_notebook_calibration():
    from convert_legacy_calibration import convert
    legacy = {'b': [.01, -.03, .02], 'L': np.eye(3),
              'aRws': np.eye(3), 'w_offset': [.02, -.05, .01],
              'w_rotation': np.eye(3)}
    result = convert(legacy)
    np.testing.assert_allclose(result.gyro_bias_dps, [.02, -.05, .01])
    with pytest.raises(ValueError, match='w_offset'):
        convert({'b': [0, 0, 0], 'L': np.eye(3)})
