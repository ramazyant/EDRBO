"""Simulate the BOPTEST test case `bestest_hydronic` (BOPTEST v0.x, IBPSA Project 1) for the
Building_Heating benchmark and fit the GP surrogates shipped in Benchmark/data/boptest_hydronic_gp.npz.

The test-case FMU (testcases/bestest_hydronic/models/wrapped.fmu of
https://github.com/ibpsa/project1-boptest) is simulated directly with FMPy instead of through the
BOPTEST REST API (identical model; the KPIs below follow BOPTEST's definitions). Linux only (the
FMU binary is linux64 and needs libgfortran.so.4, e.g. from the Ubuntu package libgfortran4).

One simulation = one day d of the heating season with the heating-setpoint schedule
    T_occ = x1 in [20.5, 22.5] degC during the occupied hours 00:00-07:00 and from (20:00 - x2) to 24:00,
    x2 in [0, 4] h = pre-heating lead time before the evening occupancy at 20:00,
    15 degC (the test case's unoccupied setpoint) otherwise,
applied for `warmup` days before d as well (periodic regime). Outputs for day d:
    cost  = gas energy * 0.0498 EUR/kWh + pump electricity * 0.2535 EUR/kWh  (test-case prices)
    tdis  = thermal discomfort in Kh: deviation of the operative zone temperature from the comfort
            band [21, 24] degC in the occupied hours and [15, 30] degC otherwise (BOPTEST bounds)
    Tout  = daily mean outdoor dry-bulb temperature (degC), Hglo = daily global horizontal
            irradiation (kWh/m2), the contexts.

    python scripts/generate_boptest_data.py --fmu /path/to/project1-boptest/testcases/bestest_hydronic/models/wrapped.fmu \
        --n 2000 --out Benchmark/data/boptest_hydronic_sims.csv
    python scripts/generate_boptest_data.py --fit Benchmark/data/boptest_hydronic_sims.csv
"""
import argparse
import os
import sys

import numpy as np

T_OCC_RANGE = (20.5, 22.5)
PREHEAT_RANGE = (0.0, 4.0)
T_UNOCC = 15.0
PRICE_GAS, PRICE_EL = 0.0498, 0.2535
STEP = 300.0  # s


def heating_season_days():
    # day-of-year (0-based start index) of days in Oct-Apr whose warm-up stays inside the year
    return np.array([d for d in range(8, 120)] + [d for d in range(273, 364)])


def setpoint_profile(x1, x2, t_day):
    """Heating setpoint (degC) averaged over [t, t+STEP) for t = seconds since midnight."""
    on_evening = 20.0 * 3600 - x2 * 3600.0
    a, b = t_day, t_day + STEP
    occ = max(0.0, min(b, 7 * 3600.0) - max(a, 0.0)) + max(0.0, min(b, 86400.0) - max(a, on_evening))
    frac = occ / STEP
    return T_UNOCC + frac * (x1 - T_UNOCC)


def comfort_bounds(t_day):
    occupied = (t_day < 7 * 3600) or (t_day >= 20 * 3600)
    return (21.0, 24.0) if occupied else (15.0, 30.0)


class FMUSim:
    def __init__(self, fmu_path):
        from fmpy import read_model_description, extract
        from fmpy.fmi2 import FMU2Slave
        self.md = read_model_description(fmu_path)
        self.dir = extract(fmu_path)
        self.vr = {v.name: v.valueReference for v in self.md.modelVariables}
        self.fmu = FMU2Slave(guid=self.md.guid, unzipDirectory=self.dir,
                             modelIdentifier=self.md.coSimulation.modelIdentifier, instanceName='bop')
        self.fmu.instantiate()
        self.fresh = True

    def run_day(self, day, x1, x2, warmup=3):
        f, vr = self.fmu, self.vr
        if not self.fresh:
            f.reset()
        self.fresh = False
        t0 = (day - warmup) * 86400.0
        f.setupExperiment(startTime=t0)
        f.enterInitializationMode()
        f.exitInitializationMode()
        f.setReal([vr['oveTSetHea_activate']], [1.0])
        outs = [vr[k] for k in ('reaTRoo_y', 'reaQHea_y', 'reaPPum_y', 'weaSta_reaWeaTDryBul_y',
                                'weaSta_reaWeaHGloHor_y')]
        t = t0
        rec = []
        prev = np.asarray(f.getReal(outs))
        n_steps = int((warmup + 1) * 86400 / STEP)
        for k in range(n_steps):
            tday = t % 86400.0
            f.setReal([vr['oveTSetHea_u']], [setpoint_profile(x1, x2, tday) + 273.15])
            f.doStep(currentCommunicationPoint=t, communicationStepSize=STEP)
            t += STEP
            cur = np.asarray(f.getReal(outs))
            if t > day * 86400.0:  # inside the evaluation day: trapezoid over [t-STEP, t]
                rec.append((tday, prev, cur))
            prev = cur
        gas = el = tdis = 0.0
        Tout, H = [], []
        for tday, a, b in rec:
            m = 0.5 * (a + b)
            Troo = m[0] - 273.15
            lo, hi = comfort_bounds(tday)
            tdis += (max(0.0, lo - Troo) + max(0.0, Troo - hi)) * STEP / 3600.0
            gas += m[1] * STEP / 3.6e6
            el += m[2] * STEP / 3.6e6
            Tout.append(m[3] - 273.15)
            H.append(m[4])
        cost = PRICE_GAS * gas + PRICE_EL * el
        return dict(cost=cost, tdis=tdis, gas_kWh=gas, el_kWh=el, Tout=float(np.mean(Tout)),
                    Hglo=float(np.sum(H) * STEP / 3.6e6))


def simulate(a):
    import pandas as pd
    from torch.quasirandom import SobolEngine
    days = heating_season_days()
    U = SobolEngine(3, scramble=True, seed=a.seed).draw(a.n).double().numpy()
    sim = FMUSim(a.fmu)
    rows = []
    for i, u in enumerate(U):
        x1 = T_OCC_RANGE[0] + u[0] * (T_OCC_RANGE[1] - T_OCC_RANGE[0])
        x2 = PREHEAT_RANGE[0] + u[1] * (PREHEAT_RANGE[1] - PREHEAT_RANGE[0])
        day = int(days[min(int(u[2] * len(days)), len(days) - 1)])
        r = sim.run_day(day, x1, x2, warmup=a.warmup)
        rows.append(dict(day=day, x1=x1, x2=x2, **r))
        if i % 100 == 0:
            print(i, rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print('wrote', a.out)


def fit(a):
    """Fit exact GPs (Matern-5/2 ARD + noise, botorch) to cost and discomfort on inputs in [0,1]^3 / ^4."""
    import pandas as pd
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.constraints import Interval, GreaterThan
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import LinearMean
    MIN_LS, MAX_LS = 0.2, 5.0  # smooth in every input (normalised units); unmodelled (e.g. day-specific) effects become noise
    df = pd.read_csv(a.fit)
    # heating months only (Nov-Mar): in Oct/Apr discomfort is dominated by solar overheating that the
    # heating schedule cannot influence
    month = pd.to_datetime(df.day, unit='D', origin='2001-01-01').dt.month
    df = df[month.isin([11, 12, 1, 2, 3])].reset_index(drop=True)
    print(f'{len(df)} simulations in Nov-Mar')
    T_LO, T_HI, H_LO, H_HI = -6.0, 14.0, 0.0, 4.0
    u1 = (df.x1 - T_OCC_RANGE[0]) / (T_OCC_RANGE[1] - T_OCC_RANGE[0])
    u2 = (df.x2 - PREHEAT_RANGE[0]) / (PREHEAT_RANGE[1] - PREHEAT_RANGE[0])
    c1 = (df.Tout - T_LO) / (T_HI - T_LO)
    c2 = (df.Hglo - H_LO) / (H_HI - H_LO)
    out = {'T_range': np.array([T_LO, T_HI]), 'H_range': np.array([H_LO, H_HI]),
           'x1_range': np.array(T_OCC_RANGE), 'x2_range': np.array(PREHEAT_RANGE)}
    for tag, cols in (('1C', [u1, u2, c1]), ('2C', [u1, u2, c1, c2])):
        X = torch.tensor(np.stack([np.asarray(c, float) for c in cols], 1))
        for target in ('cost', 'tdis'):
            y = torch.tensor(df[target].to_numpy(float)).reshape(-1, 1)
            ym, ys = y.mean(), y.std()
            gp = SingleTaskGP(X, (y - ym) / ys, outcome_transform=None,
                              covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[1],
                                                                     lengthscale_constraint=Interval(MIN_LS, MAX_LS)),
                                                        outputscale_constraint=Interval(1e-3, 20.0)),
                              likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-4)),
                              mean_module=LinearMean(X.shape[1]))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            k = gp.covar_module
            base = k.base_kernel if hasattr(k, 'base_kernel') else k
            ls = base.lengthscale.detach().numpy().ravel()
            os_ = float(k.outputscale.detach()) if hasattr(k, 'outputscale') else 1.0
            noise = float(gp.likelihood.noise.detach())
            mw = gp.mean_module.weights.detach().numpy().ravel()
            mb = float(gp.mean_module.bias.detach())
            with torch.no_grad():
                K = os_ * base(X).to_dense() + noise * torch.eye(len(X), dtype=X.dtype)
                alpha = torch.linalg.solve(K, (y - ym) / ys - gp.mean_module(X).reshape(-1, 1)).numpy().ravel()
                pred = gp.posterior(X).mean.numpy().ravel() * float(ys) + float(ym)
            resid = df[target].to_numpy(float) - pred
            ho, nte = holdout_rmse(X, (y - ym) / ys, df.day.to_numpy(), torch.tensor(ls), os_, noise, gp.mean_module)
            print(f'{tag} {target}: lengthscales {np.round(ls, 3)}, outputscale {os_:.3f}, noise {noise:.4f}, '
                  f'in-sample RMSE {np.sqrt(np.mean(resid ** 2)):.4f}, RMSE on {nte} runs of held-out days '
                  f'{ho * float(ys):.4f} (sd of data {float(ys):.4f})')
            out.update({f'{tag}_{target}_X': X.numpy(), f'{tag}_{target}_alpha': alpha,
                        f'{tag}_{target}_ls': ls, f'{tag}_{target}_os': os_, f'{tag}_{target}_mean_w': mw, f'{tag}_{target}_mean_b': mb,
                        f'{tag}_{target}_ym': float(ym), f'{tag}_{target}_ys': float(ys)})
    dest = os.path.join(os.path.dirname(os.path.abspath(a.fit)), 'boptest_hydronic_gp.npz')
    np.savez_compressed(dest, **out)
    print('wrote', dest)


def days_table(a):
    """Daily context values (Tout, Hglo) of all heating-season days from the test-case weather file."""
    import pandas as pd
    w = pd.read_csv(a.weather)
    w['day'] = (w.time // 86400).astype(int)
    g = w.groupby('day').agg(Tout=('TDryBul', 'mean'), Hglo=('HGloHor', 'mean'))
    g['Tout'] -= 273.15
    g['Hglo'] *= 24 / 1000.0  # W/m2 mean -> kWh/m2/day
    g = g.loc[heating_season_days()]
    g['month'] = pd.to_datetime(g.index, unit='D', origin='2001-01-01').month
    g.to_csv(a.out, float_format='%.4f')
    print('wrote', a.out, len(g), 'days')


def holdout_rmse(X, y, groups, ls, os_, noise, mean_fn, frac=0.1, seed=0):
    """RMSE on held-out groups (e.g. whole days) with the fitted hyperparameters (no refit)."""
    import torch
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    test_g = rng.choice(g, max(1, int(frac * len(g))), replace=False)
    te = np.isin(groups, test_g)
    Xtr, Xte = X[~te], X[te]

    def k(A, B):
        r = torch.cdist(A / ls, B / ls)
        return os_ * (1 + 5 ** 0.5 * r + 5 / 3 * r ** 2) * torch.exp(-5 ** 0.5 * r)
    with torch.no_grad():
        Ktr = k(Xtr, Xtr) + noise * torch.eye(len(Xtr), dtype=X.dtype)
        a = torch.linalg.solve(Ktr, y[~te] - mean_fn(Xtr).reshape(-1, 1))
        pred = mean_fn(Xte).reshape(-1, 1) + k(Xte, Xtr) @ a
    return float(torch.sqrt(torch.mean((pred - y[te]) ** 2))), int(te.sum())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fmu')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument('--out', default='boptest_hydronic_sims.csv')
    ap.add_argument('--fit', default=None)
    ap.add_argument('--weather', default=None, help='testcase Resources/weather.csv -> daily context table')
    a = ap.parse_args()
    if a.weather:
        days_table(a)
    elif a.fit:
        fit(a)
    else:
        simulate(a)
