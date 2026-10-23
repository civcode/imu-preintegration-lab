"""Convert the user's *trusted* original calibration pickle to documented JSON.

WARNING: pickle executes Python object-loading instructions. Only use files
created by you on a trusted machine; never process a downloaded pickle.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import numpy as np
from imu_lab.calibration import Calibration


def convert(user_data: dict) -> Calibration:
    """Map the original notebook parameter names to validated matrices.

    The notebook stores acceleration in g and gyroscope in degree/second.
    `w_rotation` maps calibrated gyro readings into the chosen body frame.
    Missing optional transforms are identity, but missing required biases
    should be investigated before navigation.
    """
    for name in ('b', 'L', 'w_offset'):
        if name not in user_data:
            raise ValueError(f"Original calibration lacks {name!r}")
    return Calibration(
        accel_bias_g=np.asarray(user_data['b'], float),
        accel_matrix=np.asarray(user_data['L'], float),
        accel_body_rotation=np.asarray(user_data.get('aRws', np.eye(3)), float),
        gyro_bias_dps=np.asarray(user_data['w_offset'], float),
        gyro_matrix=np.asarray(user_data.get('w_rotation', np.eye(3)), float),
    ).validate()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trusted_pickle', type=Path)
    p.add_argument('--out', type=Path, default=Path('data/calibration.json'))
    p.add_argument('--i-trust-this-file', action='store_true',
                   help='Required: pickle loading can execute arbitrary code')
    args = p.parse_args(argv)
    if not args.i_trust_this_file:
        p.error('Refusing to unpickle without --i-trust-this-file')
    with args.trusted_pickle.open('rb') as stream:
        params = convert(pickle.load(stream))  # trusted local file ONLY
    args.out.parent.mkdir(parents=True, exist_ok=True)
    params.to_json(args.out)
    print(f'Validated calibration written to {args.out}; confirm body-axis alignment.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted safely before completing conversion.')
        raise SystemExit(130)
