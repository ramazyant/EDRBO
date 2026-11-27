"""Download hourly wind capacity factors from Open Power System Data (OPSD) and write CSV files
usable by Wind_Commitment(data_file=..., eval_data_file=...).

OPSD time-series package, version 2020-10-06 (data up to 2020, CC-BY 4.0; cite
"Open Power System Data. 2020. Data Package Time series. Version 2020-10-06.
https://doi.org/10.25832/time_series/2020-10-06"):
    https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv
(~ 125 MB). Capacity factor columns are named <REGION>_wind_profile or
<REGION>_wind_onshore_profile / _offshore_profile (generation / capacity); if absent, the
script divides <REGION>_wind..._generation_actual by <REGION>_wind..._capacity.

Examples
    # German onshore wind, 2016 (nominal) and 2019 (evaluation)
    python Benchmark/scripts/download_opsd_wind.py --region DE --kind onshore --years 2016 --out Benchmark/data/opsd_DE_on_2016.csv
    python Benchmark/scripts/download_opsd_wind.py --region DE --kind onshore --years 2019 --out Benchmark/data/opsd_DE_on_2019.csv
    # then in Python / main.py:
    #   Wind_Commitment(data_file='Benchmark/data/opsd_DE_on_2016.csv',
    #                   eval_data_file='Benchmark/data/opsd_DE_on_2019.csv', negate=True)
    # list the available wind columns:
    python Benchmark/scripts/download_opsd_wind.py --list

If the download is blocked, fetch the CSV in a browser and pass it with --csv.
"""
import argparse
import os
import sys
import urllib.request

URL = 'https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None, help='local copy of time_series_60min_singleindex.csv')
    ap.add_argument('--cache', default=os.path.join(os.path.dirname(__file__), 'opsd_time_series_60min.csv'))
    ap.add_argument('--region', default='DE')
    ap.add_argument('--kind', default='onshore', choices=['all', 'onshore', 'offshore'])
    ap.add_argument('--years', nargs='*', type=int, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()

    import pandas as pd
    path = a.csv or a.cache
    if not os.path.exists(path):
        print(f'downloading {URL} -> {path} (~125 MB) ...')
        urllib.request.urlretrieve(URL, path)
    head = pd.read_csv(path, nrows=0).columns.tolist()
    wind_cols = [c for c in head if '_wind' in c]
    if a.list:
        print('\n'.join(wind_cols))
        return

    var = 'wind' if a.kind == 'all' else f'wind_{a.kind}'
    prof = f'{a.region}_{var}_profile'
    gen, cap = f'{a.region}_{var}_generation_actual', f'{a.region}_{var}_capacity'
    if prof in head:
        df = pd.read_csv(path, usecols=['utc_timestamp', prof])
        cf = df[prof]
    elif gen in head and cap in head:
        df = pd.read_csv(path, usecols=['utc_timestamp', gen, cap])
        cf = df[gen] / df[cap]
    else:
        sys.exit(f'no {prof} (nor {gen} and {cap}) in the file; available wind columns:\n  ' +
                 '\n  '.join(c for c in wind_cols if c.startswith(a.region + '_')) or '(none)')

    t = pd.to_datetime(df['utc_timestamp'], utc=True)
    out = pd.DataFrame({'time': t.dt.strftime('%Y-%m-%d %H:%M:%S+0000'), 'year': t.dt.year,
                        'month': t.dt.month, 'cf': cf})
    out = out[out.cf.notna() & (out.cf >= 0) & (out.cf <= 1.2)]
    out['cf'] = out.cf.clip(0, 1)
    if a.years:
        out = out[out.year.isin(a.years)]
    if out.empty:
        sys.exit('no data left after filtering (check --region/--kind/--years)')
    dest = a.out or os.path.join(os.path.dirname(__file__), '..', 'data',
                                 f'opsd_{a.region}_{a.kind}_{"-".join(map(str, a.years or ["all"]))}.csv')
    out.to_csv(dest, index=False, float_format='%.5f')
    print(f'wrote {dest}: {len(out)} hours, years {sorted(out.year.unique())}, mean CF {out.cf.mean():.3f}')


if __name__ == '__main__':
    main()
