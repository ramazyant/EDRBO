"""Expected and worst-case (Wasserstein-ball) evaluation of finished BO runs.

For every evaluated decision x_t of every run it computes, in natural units,
    E(x_t)        = E_{c ~ P_eval} f(x_t, c)                                   (expected value)
    WC_eps(x_t)   = inf_{Q : W_p(Q, P_ctr) <= eps} E_{c ~ Q} f(x_t, c)          (worst case)
and the regrets  max_x E - E(x_t)  and  max_x WC_eps - WC_eps(x_t).  Distances are measured in
the normalised context coordinates [0, 1]^{d_c} used by the optimisers.  The ball centre P_ctr
is P_eval by default (--center nominal for the data-generating distribution).

The worst case is computed exactly for distributions supported on a fine grid Z of the context
box (plus the atoms of P_ctr), through the dual of the transport LP
    WC = sup_{lam >= 0}  -lam eps^p + sum_i w_i min_{z in Z} [ f(x, z) + lam ||c_i - z||^p ],
a concave 1-D problem (bounded Brent search over lam in [0, (max f - min f) / eps^p]).

Usage (from the repository root, after main.py runs):
    python real_world_tools/robust_eval.py --problem Wake_Steering_Shift \
        --optimizers UCB SBOKDE WassersteinUCB EDRBO --seeds 100 101 102 \
        --rounds 200 --beta 1.5 --radii 0.02 0.05 0.1 --p 2 --stride 5
Writes Result/robust_<problem>_<center>_p<p>.csv (one row per optimiser, seed, iteration) and
prints a summary (final simple regret = best-so-far, and cumulative regret).
Use --test to check the dual against scipy's LP solver on random instances.
"""
import argparse
import glob
import json
import os
import pickle as pkl
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402
from torch.quasirandom import SobolEngine  # noqa: E402


# ----------------------------------------------------------------------------- core
def compress(C, w, bins):
    """Merge atoms that fall in the same cell of a regular grid (bins per axis) into their
    weighted mean; changes the distribution by at most one cell diameter in W_p."""
    C = np.asarray(C, float)
    idx = np.minimum((C * bins).astype(int), bins - 1)
    key = np.ravel_multi_index(tuple(idx.T), (bins,) * C.shape[1])
    uniq, inv = np.unique(key, return_inverse=True)
    W = np.bincount(inv, weights=w)
    Cm = np.stack([np.bincount(inv, weights=w * C[:, k]) for k in range(C.shape[1])], 1) / W[:, None]
    return Cm, W / W.sum()


def context_grid(dc, n_per_dim):
    g = np.linspace(0.0, 1.0, n_per_dim)
    if dc == 1:
        return g.reshape(-1, 1)
    G = np.meshgrid(*([g] * dc), indexing='ij')
    return np.stack([a.ravel() for a in G], 1)


class WorstCase:
    """Precomputes the transport costs between the atoms of P_ctr and the target set Z."""

    def __init__(self, C_ctr, w_ctr, Z_grid, p=2):
        self.C, self.w, self.p = C_ctr, w_ctr, p
        self.Z = np.vstack([Z_grid, C_ctr])  # include the atoms: zero-cost transport available
        self.Dp = np.linalg.norm(self.C[:, None, :] - self.Z[None, :, :], axis=-1) ** p

    def value(self, fZ, eps, rtol=1e-9):
        """fZ: objective at self.Z (natural units). Returns inf_{W_p(Q, P_ctr) <= eps} E_Q f."""
        fZ = np.asarray(fZ, float)
        f_atoms = fZ[-len(self.C):]
        if eps <= 0:
            return float(self.w @ f_atoms)
        ep = eps ** self.p

        def phi(lam):
            return -lam * ep + float(self.w @ np.min(fZ[None, :] + lam * self.Dp, axis=1))

        hi = max((fZ.max() - fZ.min()) / ep, 1e-12)
        res = minimize_scalar(lambda l: -phi(l), bounds=(0.0, hi), method='bounded',
                              options={'xatol': rtol * hi + 1e-12, 'maxiter': 200})
        return max(phi(res.x), phi(0.0), phi(hi))


def lp_check(n_tests=5, seed=0):
    """Compare the dual with the primal transport LP (scipy HiGHS) on random instances."""
    from scipy.optimize import linprog
    rng = np.random.default_rng(seed)
    worst = 0.0
    for t in range(n_tests):
        dc = 1 + t % 2
        m = 15
        C = rng.random((m, dc))
        w = rng.random(m)
        w /= w.sum()
        Z = context_grid(dc, 9)
        wc = WorstCase(C, w, Z, p=2)
        fZ = np.sin(5 * wc.Z).sum(1) + rng.normal(0, 0.1, len(wc.Z))
        eps = 0.1
        nz = len(wc.Z)
        A_eq = np.zeros((m, m * nz))
        for i in range(m):
            A_eq[i, i * nz:(i + 1) * nz] = 1
        res = linprog(np.tile(fZ, m), A_ub=wc.Dp.reshape(1, -1), b_ub=[eps ** 2], A_eq=A_eq, b_eq=w,
                      bounds=(0, None), method='highs')
        d = wc.value(fZ, eps)
        worst = max(worst, abs(res.fun - d))
        print(f'  test {t}: d_c={dc}  LP {res.fun:.8f}  dual {d:.8f}')
    print(f'max |LP - dual| = {worst:.2e}')


# ----------------------------------------------------------------------------- problem setup
def make_problem(name, robust=False):
    import Benchmark.Real_World as RW
    cls = getattr(RW, name)
    if robust and name.startswith('Wake_Steering'):
        # coarser rule for the worst case (the robust optimum uses the same rule, so regrets are
        # consistent); speeds up the transport problem
        kw = dict(n_wd_nodes=24) if '2C' not in name else dict(n_wd_nodes=16, n_ws_nodes=3)
        return cls(negate=True, E_max=0.0, E_min=0.0, **kw)
    if robust and name.startswith('Glucose_Bolus_2Meals'):
        return cls(negate=True, E_max=0.0, E_min=0.0, n_nodes=8)  # 64 atoms: each atom-target pair is a simulation
    return cls(negate=True)


def robust_objects(prob, center, p, grid_n, max_bins):
    C, w = prob.context_distribution(center)
    if len(C) > max_bins ** prob.contexts_dim // 2 or (prob.contexts_dim == 1 and len(C) > max_bins):
        C, w = compress(C, w, max_bins)
    Z = context_grid(prob.contexts_dim, grid_n)
    return WorstCase(C, w, Z, p=p)


def wc_values(prob, wc, X, radii):
    """Worst-case values for all rows of X (natural units) -> (n_x, n_radii)."""
    F = prob.f_xc(np.atleast_2d(X), wc.Z)
    return np.array([[wc.value(F[i], e) for e in radii] for i in range(len(F))])


def robust_optimum(prob, wc, radii, extra_X=None, n_sobol=128, n_local=2, seed=0):
    d = prob.decision_dim
    S = SobolEngine(d, scramble=True, seed=seed).draw(n_sobol).double().numpy()
    if extra_X is not None and len(extra_X):
        S = np.vstack([S, extra_X])
    V = np.vstack([wc_values(prob, wc, S[i:i + 32], radii) for i in range(0, len(S), 32)])
    best = []
    for k, e in enumerate(radii):
        order = np.argsort(-V[:, k])[:n_local]
        bv, bx = V[order[0], k], S[order[0]]
        for i in order:
            res = minimize(lambda z: -wc_values(prob, wc, np.clip(z, 0, 1)[None], [e])[0, 0], S[i],
                           method='Nelder-Mead', bounds=[(0, 1)] * d,
                           options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-8})
            if -res.fun > bv:
                bv, bx = -res.fun, np.clip(res.x, 0, 1)
        best.append((float(bv), bx.tolist()))
        print(f'  robust optimum eps={e:g}: {bv:.6f} at x={np.round(bx, 4)}')
    return best


def load_X(result_dir, opt, prob_name, seed, rounds, prob, beta):
    """Decisions x_t of one run, read from the result file written by main.py:
    Result/<optimizer>_<benchmark>_<seed>_<running_rounds>_<dim>_<mu>_<sigma>_<beta>_results.pkl"""
    name = f'{opt}_{prob_name}_{seed}_{rounds}_{prob.dim}_{prob.mu}_{prob.sigma}_{beta}_results.pkl'
    path = os.path.join(result_dir, name)
    if not os.path.exists(path):  # e.g. files written with other mu / sigma labels
        cands = glob.glob(os.path.join(result_dir, f'{opt}_{prob_name}_{seed}_{rounds}_*_{beta}_results.pkl'))
        if not cands:
            return None
        path = sorted(cands)[0]
        print(f'  using {os.path.basename(path)}')
    with open(path, 'rb') as fh:
        res = pkl.load(fh)
    X = np.asarray(res['X'] if isinstance(res, dict) else res, float)
    return X[:, :prob.decision_dim]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem')
    ap.add_argument('--optimizers', nargs='*', default=['UCB', 'SBOKDE', 'WassersteinUCB'])
    ap.add_argument('--seeds', nargs='*', type=int, default=list(range(100, 110)))
    ap.add_argument('--rounds', type=int, default=200)
    ap.add_argument('--beta', default='1.5')
    ap.add_argument('--radii', nargs='*', type=float, default=[0.02, 0.05, 0.1])
    ap.add_argument('--p', type=int, default=2)
    ap.add_argument('--center', default='eval', choices=['eval', 'nominal'])
    ap.add_argument('--stride', type=int, default=1, help='evaluate every stride-th iterate')
    ap.add_argument('--grid', type=int, default=None, help='grid points per context dim (default 401 / 41)')
    ap.add_argument('--max_bins', type=int, default=400, help='atom compression for P_ctr')
    ap.add_argument('--result_dir', default=os.path.join(ROOT, 'Result'))
    ap.add_argument('--test', action='store_true', help='check the dual against an LP solver')
    a = ap.parse_args()
    if a.test:
        lp_check()
        return
    beta = a.beta if '.' in a.beta else f'{float(a.beta)}'

    prob = make_problem(a.problem)                 # expected value / max_stochastic
    prob_r = make_problem(a.problem, robust=True)  # worst case
    grid_n = a.grid or (401 if prob.contexts_dim == 1 else 41)
    wc = robust_objects(prob_r, a.center, a.p, grid_n, a.max_bins if prob.contexts_dim == 1 else 40)
    print(f'{a.problem}: P_{a.center} with {len(wc.C)} atoms, target set {len(wc.Z)} points, '
          f'E_max = {prob.E_max:.6f}')

    runs = {}
    for opt in a.optimizers:
        for s in a.seeds:
            X = load_X(a.result_dir, opt, a.problem, s, a.rounds, prob, beta)
            if X is None:
                print(f'  missing: {opt} seed {s}')
                continue
            runs[(opt, s)] = X
    if not runs:
        print('no runs found')
        return

    cache = os.path.join(a.result_dir, f'robust_opt_{a.problem}_{a.center}_p{a.p}.json')
    store = json.load(open(cache)) if os.path.exists(cache) else {}
    need = [e for e in a.radii if f'{e:g}' not in store]
    if need:
        print('computing robust optima ...')
        allX = np.unique(np.round(np.vstack(list(runs.values())), 10), axis=0)
        sub = allX[np.random.default_rng(0).choice(len(allX), min(len(allX), 128), replace=False)]
        for e, (v, x) in zip(need, robust_optimum(prob_r, wc, need, extra_X=sub)):
            store[f'{e:g}'] = {'value': v, 'x': x}
        json.dump(store, open(cache, 'w'), indent=1)
    V_star = np.array([store[f'{e:g}']['value'] for e in a.radii])

    rows = []
    t0 = time.time()
    for (opt, s), X in runs.items():
        idx = np.arange(0, len(X), a.stride)
        E = prob.expected_value(X[idx], 'eval')
        W = np.vstack([wc_values(prob_r, wc, X[i:i + 1], a.radii) for i in idx])
        V_star = np.maximum(V_star, W.max(0))  # keep regrets >= 0 if an iterate beats the search
        for j, t in enumerate(idx):
            rows.append([opt, s, int(t), E[j], prob.E_max - E[j]] + list(W[j]))
        print(f'  {opt} seed {s}: {len(idx)} iterates ({time.time() - t0:.0f} s)')

    import pandas as pd
    cols = ['optimizer', 'seed', 't', 'expected_value', 'expected_regret'] + \
           [f'wc_{e:g}' for e in a.radii]
    df = pd.DataFrame(rows, columns=cols)
    for k, e in enumerate(a.radii):
        df[f'robust_regret_{e:g}'] = V_star[k] - df[f'wc_{e:g}']
    out = os.path.join(a.result_dir, f'robust_{a.problem}_{a.center}_p{a.p}.csv')
    df.to_csv(out, index=False)
    print(f'wrote {out}')

    reg_cols = ['expected_regret'] + [f'robust_regret_{e:g}' for e in a.radii]
    g = df.sort_values('t').groupby(['optimizer', 'seed'])
    simple = g[reg_cols].min().groupby('optimizer').agg(['mean', 'sem'])
    cumul = g[reg_cols].sum().mul(a.stride).groupby('optimizer').agg(['mean', 'sem'])
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', None)
    print('\nfinal simple regret (best-so-far), mean and s.e. over seeds:\n', simple.round(4))
    print(f'\ncumulative regret (sum over iterates x stride={a.stride}), mean and s.e.:\n', cumul.round(3))


if __name__ == '__main__':
    main()
