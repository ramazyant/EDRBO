"""Build the bundled hourly wind capacity-factor series Benchmark/data/wind_cf_windpowerlib_2010.csv.

Source data (MIT licence, oemof developer group) -- clone https://github.com/wind-python/windpowerlib:
  * example/weather.csv            : hourly 2010 weather for one German location (80 m wind speed used)
  * tests/oedb/power_curves.csv    : manufacturer power curves (Enercon E-82/2300 used)

The single-turbine curve is converted to a wind-farm curve by Gaussian smoothing with the
Staffell & Pfenninger (2016) standard deviation sigma(v) = 0.6 + 0.2 v (the same rule windpowerlib
implements in power_curves.smooth_power_curve), the standard way to turn a turbine curve into
an aggregate capacity factor. The output column `cf` is in [0, 1].

Usage:
    python Benchmark/scripts/make_wind_cf_windpowerlib.py --windpowerlib /path/to/windpowerlib
"""
import argparse
import os

import numpy as np
import pandas as pd


def smoothed_cf(v, curve_v, curve_p, n_quad=81):
    """Farm capacity factor at wind speeds v (Staffell-Pfenninger smoothing)."""
    rated = curve_p.max()
    sig = 0.6 + 0.2 * v
    z = np.linspace(-4, 4, n_quad)
    w = np.exp(-0.5 * z ** 2)
    w /= w.sum()
    vv = v[:, None] + sig[:, None] * z[None, :]
    pp = np.interp(vv, curve_v, curve_p, left=0.0, right=0.0)  # 0 beyond cut-out (25 m/s)
    return (pp * w[None, :]).sum(1) / rated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windpowerlib', required=True, help='path to a clone of wind-python/windpowerlib')
    ap.add_argument('--turbine', default='E-82/2300')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'data',
                                                  'wind_cf_windpowerlib_2010.csv'))
    a = ap.parse_args()

    w = pd.read_csv(os.path.join(a.windpowerlib, 'example', 'weather.csv'), header=[0, 1], index_col=0)
    v80 = w[('wind_speed', '80')].to_numpy(dtype=float)
    idx = pd.to_datetime(w.index, utc=True).tz_convert('Europe/Berlin')

    pc = pd.read_csv(os.path.join(a.windpowerlib, 'tests', 'oedb', 'power_curves.csv'))
    row = pc[pc.turbine_type == a.turbine].iloc[0].drop('turbine_type').dropna()
    curve_v = row.index.astype(float).to_numpy()
    curve_p = row.to_numpy(dtype=float)
    order = np.argsort(curve_v)
    curve_v, curve_p = curve_v[order], curve_p[order]

    cf = np.clip(smoothed_cf(v80, curve_v, curve_p), 0.0, 1.0)
    out = pd.DataFrame({'time': idx.strftime('%Y-%m-%d %H:%M:%S%z'), 'month': idx.month,
                        'wind_speed_80m': v80, 'cf': cf})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False, float_format='%.6f')
    print(f'wrote {a.out}: {len(out)} hours, mean CF {cf.mean():.3f}, '
          f'winter (Oct-Mar) {cf[np.isin(idx.month, [10, 11, 12, 1, 2, 3])].mean():.3f}, '
          f'summer (Apr-Sep) {cf[np.isin(idx.month, [4, 5, 6, 7, 8, 9])].mean():.3f}')


if __name__ == '__main__':
    main()
