# Original cube calibration → guided stages

Reference: the user-provided `ftdi-dt232h-mpu6050-05.ipynb` (not committed here:
it includes extensive generated output). The source is a **sequential interactive
experiment**, not a single independent dataset and one calibration routine.

| Source notebook cells | Physical procedure and computation | Guided notebook section |
|---|---|---|
| 11–17, 34–35 | Collect *raw* multi-orientation accelerometer data; inspect cloud | Stage 1 — raw capture |
| 37–43, 59–60 | Robust ellipsoid least squares: bias `b`, lower-triangular `L`, gravity magnitude 1g | Stage 1 — fit, visualization, accept |
| 62–67 | Reconfigure acquisition to apply `(a-b)` and `L` for next experiment | Stage 2 — applies accepted ellipsoid before face fitting |
| 78–95 | Identify stationary `||w||` below chosen limit (source example 3.7°/s), class by orientation cosine >0.9, estimate six means | Stage 2 — explicit six physical labels and separate raw captures; stillness selection and six means |
| 98–104 | SVD/Kabsch rotation `aRws` into chosen orthogonal frame; compose with previous accel parameters | Stage 2 — Kabsch + per-face angular residual; checkpoint |
| 114–122 | Stationary gyro capture and offset `w_offset` | Stage 3 — uses corrected acceleration to check static-gravity consistency |
| 125–135 | Apply the earlier acceleration calibration and gyro offset during rotation capture | Stage 4 — apply earlier parameters **offline** to retained raw data before selection; mathematically same transformation before analysis |
| 136–146 | Gyro selection for ±X,±Y,±Z; source trial rate 200–240°/s, subordinate <10°/s, direction cosine >0.9; inspect selected counts/distribution | Stage 4 — **user-selectable** rate bands and cross-axis/cosine gates, first plot actual per-turn measurements |
| 140, 160–165 | Sign-flip negative turns, combine signed rates, normalize columns `U=[ux,uy,uz]`, invert `T=U⁻¹` | Stage 4 — signed axis-alignment fit, report sample counts and ± consistency |
| 171–199 | Apply all transforms and plot calibration output / persist to pickle | Stage 5 — independent stationary validation, calibrated gravity/gyro/path plot, JSON export with provenance |

## Deliberate differences

- **Unmodified raw captures are retained** in CSV and transformed only during
  analysis; the original notebook used callbacks to apply partial calibration
  during later recordings. Both give the same inputs to the mathematical
  estimators, provided every stage applies the accepted prior corrections
  **before selection/fitting**. Raw capture preserves the ability to retune
  gates without losing data.
- Six-face and signed-axis trials now get **explicit physical labels** and
  are persisted immediately; original notebook often grouped all samples
  after one longer capture using direction cosine.
- Thresholds remain experimental controls. In particular, 200–240°/s is
  not automatically correct for a given sensor setup/cube rotation.
- Raw data and predecessor checkpoint hashes are verified before downstream
  operations, preventing reuse of obsolete partial calibrations.
- None of these controlled turns provides an independent measurement of
  angular-rate scale. The fitted `T` addresses axis misalignment.
- No hardware-specific precision or full navigation performance is claimed
  without repeatable physical validation.
