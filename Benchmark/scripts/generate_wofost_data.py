"""Simulate WOFOST 8.1 (PCSE, N- and water-limited, classic water + N balance) for the
Crop_Management benchmark and fit the GP surrogates in Benchmark/data/wofost_potato_gp.npz.

Inputs (all open source):
  * PCSE >= 6.0 (Wageningen, EUPL) -- pip install pcse
  * crop parameters: branch `wofost81` of https://github.com/ajwdewit/WOFOST_crop_parameters (EUPL)
  * weather + soil: https://github.com/ajwdewit/pcse_notebooks (MIT): data/meteo/nl1.xlsx (Wageningen
    2004-2008), data/meteo/weather_wageningen.xlsx (Wageningen 2013-2015), data/soil/ec1.soil

Season: potato (Fontane), planted 20 April, harvested 20 September, sandy soil (EC1), Wageningen.
Decision x = (N, I): total mineral N fertiliser N in [0, 250] kg/ha (60 % at planting, 40 % 40 days
later, recovery 0.7) and total irrigation I in [0, 40] cm (10 equal weekly doses from 1 June).
Context c = weather perturbation of the base season: rainfall multiplier rho in [0.2, 1.8] (all
daily rain amounts scaled) and, for the 2-context version, a temperature offset dT in [-2, 4] degC
(TMIN/TMAX shifted; E0/ES0/ET0 recomputed with PCSE's Penman routine).
Output: tuber dry matter TWSO (kg/ha) at harvest; profit is computed in the benchmark class.

    python Benchmark/scripts/generate_wofost_data.py --notebooks /path/pcse_notebooks --crop /path/WOFOST_crop_parameters_wofost81 \
        --n 2500 --out wofost_potato_sims.csv
    python Benchmark/scripts/generate_wofost_data.py --fit wofost_potato_sims.csv
    python Benchmark/scripts/generate_wofost_data.py --notebooks ... --rain_table Benchmark/data/wageningen_season_rain.csv
"""
import argparse
import datetime as dt
import os

import numpy as np

N_RANGE = (0.0, 250.0)
I_RANGE = (0.0, 40.0)
RHO_RANGE = (0.2, 1.8)
DT_RANGE = (-2.0, 4.0)
BASE_YEAR = 2006
PLANT, HARVEST = (4, 20), (9, 20)


def make_weather(notebooks):
    from pcse.input import ExcelWeatherDataProvider
    import warnings
    warnings.filterwarnings('ignore')
    return [ExcelWeatherDataProvider(os.path.join(notebooks, 'data', 'meteo', f), missing_snow_depth=0, force_reload=True)
            for f in ('nl1.xlsx', 'weather_wageningen.xlsx')]


from pcse.base import WeatherDataProvider  # noqa: E402


class PerturbedWeather(WeatherDataProvider):
    """Copy of a PCSE weather provider for 1 March - 31 October of `year` with rain * rho and
    temperatures + dT; E0/ES0/ET0 recomputed with PCSE's Penman routine (reference_ET)."""

    def __init__(self, base, year, rho=1.0, dT=0.0):
        from pcse.base import WeatherDataContainer
        from pcse.util import reference_ET
        WeatherDataProvider.__init__(self)
        self.description = [f'{base.description[0] if base.description else ""} perturbed rho={rho:.3f} dT={dT:.3f}']
        self.latitude, self.longitude, self.elevation = base.latitude, base.longitude, base.elevation
        self.angstA, self.angstB = base.angstA, base.angstB
        day = dt.date(year, 3, 1)
        while day <= dt.date(year, 10, 31):
            w = base(day)
            tmin, tmax = w.TMIN + dT, w.TMAX + dT
            if dT != 0.0:
                E0, ES0, ET0 = reference_ET(DAY=w.DAY, LAT=w.LAT, ELEV=w.ELEV, TMIN=tmin, TMAX=tmax, IRRAD=w.IRRAD,
                                            VAP=w.VAP, WIND=w.WIND, ANGSTA=self.angstA, ANGSTB=self.angstB)
                E0, ES0, ET0 = E0 / 10.0, ES0 / 10.0, ET0 / 10.0  # mm/day -> cm/day
            else:
                E0, ES0, ET0 = w.E0, w.ES0, w.ET0
            wdc = WeatherDataContainer(LAT=w.LAT, LON=w.LON, ELEV=w.ELEV, DAY=w.DAY, IRRAD=w.IRRAD, TMIN=tmin,
                                       TMAX=tmax, VAP=w.VAP, RAIN=w.RAIN * rho, WIND=w.WIND, E0=E0, ES0=ES0,
                                       ET0=ET0, SNOWDEPTH=getattr(w, 'SNOWDEPTH', 0.0) or 0.0)
            self._store_WeatherDataContainer(wdc, day)
            day += dt.timedelta(days=1)


def agromanagement(year, N, I):
    plant = dt.date(year, *PLANT)
    harvest = dt.date(year, *HARVEST)
    events_n = [{plant: {'N_amount': 0.6 * N, 'N_recovery': 0.7}},
                {plant + dt.timedelta(days=40): {'N_amount': 0.4 * N, 'N_recovery': 0.7}}]
    first_irr = dt.date(year, 6, 1)
    events_i = [{first_irr + dt.timedelta(days=7 * k): {'amount': I / 10.0, 'efficiency': 0.8}} for k in range(10)]
    return [{plant - dt.timedelta(days=10): {
        'CropCalendar': {'crop_name': 'potato', 'variety_name': 'Fontane', 'crop_start_date': plant,
                         'crop_start_type': 'sowing', 'crop_end_date': harvest, 'crop_end_type': 'harvest',
                         'max_duration': 300},
        'TimedEvents': [{'event_signal': 'apply_n', 'name': 'N fertiliser', 'comment': 'kg N/ha',
                         'events_table': events_n},
                        {'event_signal': 'irrigate', 'name': 'irrigation', 'comment': 'cm',
                         'events_table': events_i}],
        'StateEvents': None}}]


class Wofost:
    def __init__(self, notebooks, crop_dir):
        from pcse.input import YAMLCropDataProvider, CABOFileReader, WOFOST81SiteDataProvider_Classic
        from pcse.models import Wofost81_NWLP_CWB_CNB
        self.model_cls = Wofost81_NWLP_CWB_CNB
        self.crop = YAMLCropDataProvider(Wofost81_NWLP_CWB_CNB, fpath=crop_dir)
        self.soil = CABOFileReader(os.path.join(notebooks, 'data', 'soil', 'ec1.soil'))
        self.site = WOFOST81SiteDataProvider_Classic(WAV=10.0, CO2=400.0, NAVAILI=10.0, NSOILBASE=40.0,
                                                     NSOILBASE_FR=0.025, BG_N_SUPPLY=0.05)
        self.weather = {}
        for wp in make_weather(notebooks):
            for y in range(wp.first_date.year, wp.last_date.year + 1):
                if wp.first_date <= dt.date(y, 4, 1) and dt.date(y, 10, 1) <= wp.last_date:
                    self.weather[y] = wp

    def run(self, N, I, rho=1.0, dT=0.0, year=BASE_YEAR):
        from pcse.base import ParameterProvider
        params = ParameterProvider(cropdata=self.crop, soildata=self.soil, sitedata=self.site)
        wdp = PerturbedWeather(self.weather[year], year, rho, dT)
        m = self.model_cls(params, wdp, agromanagement(year, N, I))
        m.run_till_terminate()
        s = m.get_summary_output()[-1]
        return float(s['TWSO'] or 0.0), s


def season_rain(a):
    import pandas as pd
    rows = []
    for wp in make_weather(a.notebooks):
        for y in range(wp.first_date.year, wp.last_date.year + 1):
            if wp.first_date <= dt.date(y, 4, 20) and dt.date(y, 9, 20) <= wp.last_date:
                days = [dt.date(y, 4, 20) + dt.timedelta(days=k) for k in range((dt.date(y, 9, 20) - dt.date(y, 4, 20)).days + 1)]
                rain = sum(wp(d).RAIN for d in days)
                temp = np.mean([(wp(d).TMIN + wp(d).TMAX) / 2 for d in days])
                rows.append(dict(year=y, rain_cm=rain, tmean=temp))
    df = pd.DataFrame(rows)
    df['rho_vs_base'] = df.rain_cm / float(df.loc[df.year == BASE_YEAR, 'rain_cm'].iloc[0])
    df.to_csv(a.rain_table, index=False, float_format='%.4f')
    print(df)


def simulate(a):
    import pandas as pd
    from torch.quasirandom import SobolEngine
    w = Wofost(a.notebooks, a.crop)
    U = SobolEngine(4, scramble=True, seed=a.seed).draw(a.n).double().numpy()
    rows = []
    for i, u in enumerate(U):
        N = N_RANGE[0] + u[0] * (N_RANGE[1] - N_RANGE[0])
        I = I_RANGE[0] + u[1] * (I_RANGE[1] - I_RANGE[0])
        rho = RHO_RANGE[0] + u[2] * (RHO_RANGE[1] - RHO_RANGE[0])
        dT = DT_RANGE[0] + u[3] * (DT_RANGE[1] - DT_RANGE[0])
        twso, _ = w.run(N, I, rho, dT)
        # 1-context design: same Sobol points with dT = 0 (separate column set)
        twso0, _ = w.run(N, I, rho, 0.0)
        rows.append(dict(N=N, I=I, rho=rho, dT=dT, TWSO=twso, TWSO_dT0=twso0))
        if i % 100 == 0:
            print(i, rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print('wrote', a.out)


def fit(a):
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
    u = lambda v, r: (np.asarray(v, float) - r[0]) / (r[1] - r[0])
    out = {'N_range': np.array(N_RANGE), 'I_range': np.array(I_RANGE), 'rho_range': np.array(RHO_RANGE),
           'dT_range': np.array(DT_RANGE)}
    for tag, cols, target in (('1C', [u(df.N, N_RANGE), u(df.I, I_RANGE), u(df.rho, RHO_RANGE)], 'TWSO_dT0'),
                              ('2C', [u(df.N, N_RANGE), u(df.I, I_RANGE), u(df.rho, RHO_RANGE), u(df.dT, DT_RANGE)], 'TWSO')):
        X = torch.tensor(np.stack(cols, 1))
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
        ho, nte = holdout_rmse(X, (y - ym) / ys, np.arange(len(df)), torch.tensor(ls), os_, noise, gp.mean_module)
        print(f'{tag}: lengthscales {np.round(ls, 3)}, outputscale {os_:.3f}, noise {noise:.5f}, '
              f'in-sample RMSE {np.sqrt(np.mean(resid ** 2)):.1f} kg/ha, held-out RMSE ({nte} runs) '
              f'{ho * float(ys):.1f} kg/ha (sd {float(ys):.1f})')
        out.update({f'{tag}_X': X.numpy(), f'{tag}_alpha': alpha, f'{tag}_ls': ls, f'{tag}_os': os_,
                    f'{tag}_mean_w': mw, f'{tag}_mean_b': mb, f'{tag}_ym': float(ym), f'{tag}_ys': float(ys)})
    dest = os.path.join(os.path.dirname(os.path.abspath(a.fit)), 'wofost_potato_gp.npz')
    np.savez_compressed(dest, **out)
    print('wrote', dest)


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
    ap.add_argument('--notebooks')
    ap.add_argument('--crop')
    ap.add_argument('--n', type=int, default=2500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='wofost_potato_sims.csv')
    ap.add_argument('--fit', default=None)
    ap.add_argument('--rain_table', default=None)
    a = ap.parse_args()
    if a.rain_table:
        season_rain(a)
    elif a.fit:
        fit(a)
    else:
        simulate(a)
