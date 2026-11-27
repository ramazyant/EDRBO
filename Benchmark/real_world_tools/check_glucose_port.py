"""Compare the vectorised patient model (Benchmark/glucose_sim.py) with simglucose itself.

    pip install simglucose ; python Benchmark/real_world_tools/check_glucose_port.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from Benchmark.glucose_sim import load_patient, simulate, simulate_schedule  # noqa: E402


def simglucose_run(name, meals, boluses, t_start=-60, t_end=360):
    """meals: list of (time, g); boluses: list of (time, U); integer times in minutes."""
    from simglucose.patient.t1dpatient import T1DPatient, Action
    pat = T1DPatient.withName(name)
    basal = pat._params.u2ss * pat._params.BW / 6000.0
    bg = []
    for k in range(int(t_end - t_start)):
        t = t_start + k
        meal = sum(g for (tm, g) in meals if tm == t)  # announced at its time, eaten at 5 g/min
        ins = basal + sum(u for (tb, u) in boluses if tb <= t < tb + 1)
        pat.step(Action(CHO=meal, insulin=ins))
        bg.append(pat.state[3] / pat._params.Vg)
    return np.asarray(bg)


def main():
    import warnings
    warnings.filterwarnings('ignore')
    worst = 0.0
    for name in ['adult#001', 'adolescent#003', 'child#002']:
        P = load_patient(name)
        cases = [(70, 7, -15), (45, 3, 0), (120, 12, 10), (80, 0, 0), (30, 6, -30)]
        t0 = time.time()
        t, BG = simulate(P, [c[0] for c in cases], [c[1] for c in cases], [c[2] for c in cases])
        dt = time.time() - t0
        for i, (c, u, tb) in enumerate(cases):
            ref = simglucose_run(name, [(0, c)], [(tb, u)])
            err = np.abs(BG[i] - ref).max()
            worst = max(worst, err)
            print(f'{name:15s} meal {c:3d} g, bolus {u:2d} U at {tb:+3d} min: max |BG_port - BG_simglucose| = '
                  f'{err:.3f} mg/dL (BG range {ref.min():.0f}-{ref.max():.0f})')
        print(f'  (port: {len(cases)} scenarios in {dt:.2f} s)')
        # two meals (breakfast at 0, lunch at 300 min), boluses 15 min before each
        for (c1, c2, u1, u2) in [(45, 70, 5, 8), (60, 90, 9, 3)]:
            t, BG2 = simulate_schedule(P, [0.0, 300.0], [[c1, c2]], [[-15.0, 285.0]], [[u1, u2]], t_start=-60, t_end=660)
            ref = simglucose_run(name, [(0, c1), (300, c2)], [(-15, u1), (285, u2)], t_start=-60, t_end=660)
            err = np.abs(BG2[0] - ref).max()
            worst = max(worst, err)
            print(f'{name:15s} two meals {c1}/{c2} g, boluses {u1}/{u2} U: max |BG_port - BG_simglucose| = {err:.3f} mg/dL')
    print(f'worst error {worst:.3f} mg/dL')


if __name__ == '__main__':
    main()
