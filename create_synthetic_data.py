"""Generate repeatable square-path test recordings (clean, noisy) + ground truth."""
from pathlib import Path
import numpy as np
import pandas as pd
from imu_lab import GRAVITY, make_synthetic

def main():
    DATA = Path(__file__).resolve().parent / 'data'
    DATA.mkdir(exist_ok=True)
    for name, params in [
    ('synthetic_ideal', {'accel_bias': (0, 0, 0), 'gyro_bias': (0, 0, 0),
                         'accel_noise_std': 0, 'gyro_noise_std': 0}),
    ('synthetic_noisy', {}),
]:
        imu, truth = make_synthetic(**params)
        df = pd.DataFrame(np.column_stack([imu.t, imu.acc, imu.gyro]),
                      columns=['t_s', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        df.to_csv(DATA / f'{name}.csv', index=False, float_format='%.9f')
        if name == 'synthetic_ideal':
            pd.DataFrame({'t_s': imu.t, 'x': truth.p[:, 0], 'y': truth.p[:, 1],
                      'z': truth.p[:, 2]}).to_csv(DATA / 'synthetic_truth.csv',
                                                       index=False, float_format='%.9f')
        print(f'Wrote {DATA / (name + ".csv")} ({len(df)} rows)')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C; no background process remains.")
        raise SystemExit(130)
