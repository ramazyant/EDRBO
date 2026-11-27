"""Compute max_x E_{P_eval} f(x, c) (and the min, needed with --minimization) for the
real-world benchmarks, i.e. the `max_stochastic` used for the stochastic regret.

    python Benchmark/real_world_tools/compute_max_stochastic.py                       # all problems, print only
    python Benchmark/real_world_tools/compute_max_stochastic.py --problem Wake_Steering --write
    python Benchmark/real_world_tools/compute_max_stochastic.py --n_sobol 2048 --n_local 8 --write

--write stores the values in the PRECOMPUTED table of Benchmark/Real_World.py, so that default
instances do not recompute them at construction. Wind_Commitment* are always solved exactly at
construction (closed form), so they are skipped here.
"""
import argparse
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import Benchmark.Real_World as RW  # noqa: E402

DEFAULT = ['Wake_Steering', 'Wake_Steering_Shift', 'Wake_Steering_2C', 'Wake_Steering_2C_Shift',
           'Robot_Push', 'Robot_Push_Shift', 'Robot_Push_Success', 'Robot_Push_Success_Shift',
           'Plant_Leaf', 'Plant_Leaf_Shift', 'Glucose_Bolus', 'Glucose_Bolus_Shift',
           'Glucose_Bolus_2Meals', 'Glucose_Bolus_2Meals_Shift',
           'Building_Heating', 'Building_Heating_Shift', 'Building_Heating_2C', 'Building_Heating_2C_Shift',
           'Crop_Management', 'Crop_Management_Shift', 'Crop_Management_2C', 'Crop_Management_2C_Shift']


def write_precomputed(table):
    path = os.path.join(ROOT, 'Benchmark', 'Real_World.py')
    src = open(path).read()
    m = re.search(r'PRECOMPUTED = \{\n(.*?)\n\}', src, flags=re.S)
    existing = {}
    for line in m.group(1).splitlines():
        mm = re.match(r"\s*'(.+)': \(([-0-9.eE+]+), ([-0-9.eE+]+)\),", line)
        if mm:
            existing[mm.group(1)] = (float(mm.group(2)), float(mm.group(3)))
    existing.update(table)
    body = '\n'.join(f"    '{k}': ({v[0]!r}, {v[1]!r})," for k, v in sorted(existing.items()))
    src = src[:m.start()] + 'PRECOMPUTED = {\n' + body + '\n}' + src[m.end():]
    open(path, 'w').write(src)
    print(f'updated PRECOMPUTED in {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', nargs='*', default=DEFAULT)
    ap.add_argument('--n_sobol', type=int, default=None, help='screening points (default per class)')
    ap.add_argument('--n_local', type=int, default=None, help='local Nelder-Mead restarts')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    table = {}
    for name in a.problem:
        prob = getattr(RW, name)(negate=True, E_max=0.0, E_min=0.0)
        kw = {}
        if a.n_sobol:
            kw['n_sobol'] = a.n_sobol
        if a.n_local:
            kw['n_local'] = a.n_local
        t = time.time()
        print(f'{name}:')
        res = prob.optimise_expectation('eval', verbose=True, **kw)
        key = prob._config_key() or name
        table[key] = (float(res['max']), float(res['min']))
        print(f'  -> key {key}: E_max = {res["max"]:.6f}, E_min = {res["min"]:.6f}  ({time.time() - t:.0f} s)')
    if a.write:
        write_precomputed(table)


if __name__ == '__main__':
    main()
