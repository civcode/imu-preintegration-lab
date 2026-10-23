# IMU preintegration lab — FT232H + MPU-6050

A **standalone, educational** Python/JupyterLab lab. Record calibrated 6-axis
MPU-6050 data through an **FT232H USB-to-I²C** interface (PyFtdi), verify
stationary sensor behavior, and compare direct inertial integration against
interval-based IMU preintegration. No Arduino and no visual-odometry dependency.

The sensor-only 3D path **will drift**. Preintegration does not remove drift:
it packages motion constraints over an interval for a future estimator.

## Start without hardware

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q                  # no sensor required
jupyter lab
```

From the repository root, open these notebooks in order:

| Notebook | Main idea | Hardware needed? |
|---|---|---|
| `01_calibration.ipynb` | Ellipsoid, six-pose alignment, gyro bias & axis mapping (synthetic demo) | No for demo; yes for your calibration |
| `02_ftdi_record.ipynb` | USB/I²C, FIFO sample clock and overflow handling | Yes |
| `03_stationary_validation.ipynb` | 20-second stillness check, false drift visualization | Real recording |
| `04_preintegration.ipynb` | Direct vs interval preintegration, 3D path | No for synthetic demo |

All notebook code cells catch `KeyboardInterrupt` and let the kernel continue.
Recording uses a `with` block that closes the I²C controller even on Ctrl+C.
CLI commands likewise exit cleanly with code 130 on Ctrl+C. The captured
partial recording is saved when the interrupt happens **inside** `capture()`.

## Sensor hardware and recording

Use the wiring and FTDI driver setup from your working `ftdi-dt232h-mpu6050-05.ipynb`.
The example assumes `ftdi://ftdi:232h/1`, MPU-6050 I²C address 0x68 and
400 kHz bus. Check your breakout's supply voltage and logic levels.

Configured full scales: ±2g accelerometer, ±250°/s gyroscope. With
`DLPF_CFG=3` and `SMPLRT_DIV=4`, the nominal sample frequency is 200 Hz.

```bash
# For an initial raw stationary test without calibration:
python record_ftdi.py --seconds 20 --out data/stationary_raw.csv

# Once calibration.json contains your REAL parameters:
python record_ftdi.py --seconds 20 \
  --calibration data/calibration.json --out data/stationary.csv

python validate_stationary.py data/stationary.csv --plot data/stationary_drift.png
```

The recorder produces two files: `stationary.csv` (numeric SI measurements)
and `stationary.metadata.json` (nominal sample frequency, timestamp basis,
stop reason, FIFO statistics, calibration flag, detected overflow). The CSV
columns are `t_s,ax,ay,az,gx,gy,gz,temp_c`: acceleration is **specific force
[m/s²]** (approximately +9.80665 m/s² on body +Z when level and still), gyro
is **rad/s**; `temp_c` is optional. Accel and gyro must share body axes.

### Why the timestamps are different from the original notebook

The MPU-6050 batches 12-byte accel/gyro samples in FIFO **before** USB reads
them. Giving every decoded packet `time.time()` measures Python processing,
not acquisition. The recorder resets FIFO when capture starts and assigns:

```python
t_s = sample_index / sample_rate_hz
```

This is **nominal sensor time**. It assumes no missed samples and a stable
sample-rate generator. The FIFO does not provide hardware timestamps, and
its clock-frequency error cannot be measured from individual FIFO entries.
`elapsed_host_s` is diagnostic only, not a substitute for sample timestamps.
On latched FIFO overflow, acquisition stops, metadata sets
`overflow_detected=true`, and the integration tools **refuse the recording**.
Sample continuity cannot be recovered by silently resetting the FIFO.

A 1024-byte FIFO holds only ~85 12-byte samples (~0.425 s at 200 Hz).
Avoid slow polling, logging every sample to the console, or long sleeps
while capture is running. For genuine hardware-loss diagnosis you may need
an externally synchronized clock or additional acquisition hardware.

### Calibrating from your original notebook

Your notebook's model is preserved in `imu_lab/calibration.py`:

```text
acc_cal_g = accel_body_rotation @ accel_matrix @ (acc_raw_g - accel_bias_g)
gyro_cal_dps = gyro_matrix @ (gyro_raw_dps - gyro_bias_dps)
```

Key names from your original `user_data` pickle (same expected raw units):

| Original key | JSON key |
|---|---|
| `b` | `accel_bias_g` |
| `L` | `accel_matrix` |
| `aRws` | `accel_body_rotation` |
| `w_offset` | `gyro_bias_dps` |
| `w_rotation` | `gyro_matrix` |

Use notebook 01 to learn how these are estimated. The included
`examples/calibration.example.json` is **identity/zero and not a measured
calibration**. Do not run real-motion experiments with synthetic demo values.
Six-pose calibration requires **explicit labels** for which side of the
sensor points upward in each static pose. Gyro-axis calibration from three
independent turns adjusts alignment but does **not** provide a trustworthy
angular-rate scale-factor measurement. Always check that corrected acceleration
and gyro are in the SAME right-handed body frame and apply calibration ONCE.

## What preintegration teaches

`imu_lab/core.py` uses world +Z up, gravity `g_W=[0,0,-9.80665]` m/s²,
and `R_WB` as body-to-world orientation. A sample's gyro and specific
force are held constant over `[t_k,t_(k+1))`. Direct integration computes:

```text
R_next = R * Exp((gyro - gyro_bias)*dt)
a_world = R_mid * (specific_force - accel_bias) + gravity_world
v_next = v + a_world*dt
p_next = p + v*dt + 0.5*a_world*dt^2
```

Preintegration instead accumulates `ΔR`, `Δv`, `Δp` relative to the **starting
body frame** of each keyframe interval, without gravity. At the interval end:

```text
R_j = R_i * ΔR
v_j = v_i + g_W*Δt + R_i*Δv
p_j = p_i + v_i*Δt + 0.5*g_W*Δt² + R_i*Δp
```

Both paths must agree at keyframe times, up to numerical precision.
The implementation intentionally omits covariance propagation, bias Jacobians,
optimization, gravity refinement, magnetometer or ZUPT corrections. An MPU-6050
alone cannot determine absolute yaw from gravity; position and velocity drift.

## Suggested sequence of experiments

1. Execute the synthetic notebook and verify direct vs preintegrated agreement.
2. Capture 20 seconds of **zero motion**; check gyro residual, specific-force
   magnitude and spurious integrated displacement.
3. Make a short 1D displacement with fixed sensor orientation, starting/ending still.
4. Rotate in place. Observe how attitude error leaks gravity into translation.
5. Change preintegration interval sizes; compare their numerical agreement.

### Validation and known limitations

Run tests locally: `python -m pytest -q`. Hardware-dependent USB captures
cannot be verified without your FT232H + MPU-6050. Synthetic tests and a fake
I²C FIFO device validate timestamps, overflow stopping and Ctrl+C cleanup.
No GitHub Actions run or automatic workflow triggers are configured.

References: [MPU-6000/6050 Register Map](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Register-Map1.pdf),
[Forster et al. (2015), On-Manifold Preintegration for Real-Time Visual-Inertial Odometry](https://arxiv.org/abs/1512.02363).
