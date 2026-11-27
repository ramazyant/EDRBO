"""Summary of the real-world benchmarks: dimensions, optimum, and size of the distribution shift.

For every problem it prints d_x, d_c, max_x E_eval f, and for the *_Shift problems
  W1, W2   : Wasserstein distances between P_nominal and P_eval (normalised context coordinates)
  shift regret : E_eval-regret of the decision that is optimal under P_nominal
                 (what a learner that models P_nominal perfectly would lose).

    python real_world_tools/describe_benchmarks.py [--problem Wake_Steering_Shift ...]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from real_world_tools.robust_eval import compress  # noqa: E402

ALL = ['Wind_Commitment', 'Wind_Commitment_Shift', 'Wake_Steering', 'Wake_Steering_Shift',
       'Wake_Steering_2C', 'Wake_Steering_2C_Shift', 'Robot_Push', 'Robot_Push_Shift',
       'Robot_Push_Success', 'Robot_Push_Success_Shift', 'Plant_Leaf', 'Plant_Leaf_Shift',
       'Glucose_Bolus', 'Glucose_Bolus_Shift', 'Glucose_Bolus_2Meals', 'Glucose_Bolus_2Meals_Shift',
       'Building_Heating', 'Building_Heating_Shift', 'Building_Heating_2C', 'Building_Heating_2C_Shift',
       'Crop_Management', 'Crop_Management_Shift', 'Crop_Management_2C', 'Crop_Management_2C_Shift']


def wasserstein(C1, w1, C2, w2, p):
    if C1.shape[1] == 1:  # exact, via quantile functions
        a, b = C1[:, 0], C2[:, 0]
        ia, ib = np.argsort(a), np.argsort(b)
        a, wa, b, wb = a[ia], w1[ia], b[ib], w2[ib]
        u = np.unique(np.concatenate([np.cumsum(wa), np.cumsum(wb), [0.0]]).clip(0, 1))
        mid = (u[1:] + u[:-1]) / 2
        qa = a[np.minimum(np.searchsorted(np.cumsum(wa), mid), len(a) - 1)]
        qb = b[np.minimum(np.searchsorted(np.cumsum(wb), mid), len(b) - 1)]
        return float((np.diff(u) @ np.abs(qa - qb) ** p) ** (1 / p))
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, vstack
    C1, w1 = compress(C1, w1, 24)
    C2, w2 = compress(C2, w2, 24)
    m, n = len(C1), len(C2)
    cost = (np.linalg.norm(C1[:, None] - C2[None], axis=-1) ** p).ravel()
    rows = np.repeat(np.arange(m), n)
    cols = np.arange(m * n)
    A1 = coo_matrix((np.ones(m * n), (rows, cols)), shape=(m, m * n))
    A2 = coo_matrix((np.ones(m * n), (np.tile(np.arange(n), m), cols)), shape=(n, m * n))
    # the two marginal constraints are linearly dependent: drop one row to avoid spurious infeasibility
    A = vstack([A1, A2]).tocsr()[:-1]
    # (masses scaled by 1e3: HiGHS' feasibility tolerance is otherwise too coarse for tiny atoms)
    b = 1e3 * np.concatenate([w1, w2 * (w1.sum() / w2.sum())])[:-1]
    res = linprog(cost, A_eq=A, b_eq=b, bounds=(0, None), method='highs')
    if res.status != 0:
        raise RuntimeError(res.message)
    return float((res.fun / 1e3) ** (1 / p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', nargs='*', default=ALL)
    a = ap.parse_args()
    import Benchmark.Real_World as RW
    print(f'{"problem":28s} {"d_x":>3s} {"d_c":>3s} {"E_max (eval)":>13s} {"W1":>7s} {"W2":>7s} {"shift regret":>13s}')
    for name in a.problem:
        prob = getattr(RW, name)(negate=True)
        line = f'{name:28s} {prob.decision_dim:3d} {prob.contexts_dim:3d} {prob.E_max:13.5f}'
        if name.endswith('Shift'):
            Cn, wn = prob.context_distribution('nominal')
            Ce, we = prob.context_distribution('eval')
            W1, W2 = wasserstein(Cn, wn, Ce, we, 1), wasserstein(Cn, wn, Ce, we, 2)
            x_nom = prob.optimise_expectation('nominal')['argmax']
            reg = prob.E_max - prob.expected_value(np.atleast_2d(x_nom), 'eval')[0]
            line += f' {W1:7.4f} {W2:7.4f} {reg:13.5f}'
        print(line, flush=True)


if __name__ == '__main__':
    main()
