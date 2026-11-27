"""Real-world continuous-context benchmarks for (distributionally robust) contextual BO.

Every class follows the interface of the synthetic test problems in Benchmark/Test_Function.py
(the same interface as in WDRBO / SBOKDE), so it runs with main.py and every optimiser in
edrbo/optimizers.py, and produces the same result files (see Benchmark/README_real_world.md):

  * botorch ``SyntheticTestFunction`` on the unit cube [0, 1]^dim; the LAST ``contexts_dim``
    coordinates are the context, the first ``dim - contexts_dim`` the decision;
  * ``evaluate_true(X)`` takes an ``n x (dim - contexts_dim)`` decision tensor, draws the context
    from the *nominal* (data-generating) distribution and returns ``(f, contexts)``;
  * ``evaluate_stochastic(X)`` returns E_{c ~ P_eval} f(x, c) (deterministic quadrature / exact
    average over data), used for the stochastic regret together with ``max_stochastic``;
  * sign convention as in ``Continuous_Vendor``: the natural objective (profit, power gain,
    push reward) is *maximised* when ``negate=True``, which is what ``main.py`` passes by
    default (``negate = not args.minimization``);
  * ``mu`` / ``sigma`` exist only because ``main.py`` puts them in the result file names.

For the ``*_Shift`` classes the optimiser observes contexts from P_nominal while regret is
measured under P_eval != P_nominal (``max_stochastic`` is the optimum under P_eval). For the
other classes P_nominal = P_eval (the usual data-driven setting).

Extra methods used by ``Benchmark/real_world_tools/robust_eval.py`` and ``Benchmark/real_world_tools/compute_max_stochastic.py``:
  * ``f_xc(X, C)``      -> numpy (n_x, n_c): natural objective on all pairs (x_i, c_j);
  * ``context_distribution(which)`` -> (C, w): discrete representation of P_nominal / P_eval
    (atoms + weights) used for expectations and for the Wasserstein worst case;
  * ``expected_value(X, which)`` -> numpy (n_x,): E f in natural units.
"""
import math
import os

import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch.quasirandom import SobolEngine

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

__all__ = ['Wind_Commitment', 'Wind_Commitment_Shift',
           'Wake_Steering', 'Wake_Steering_Shift', 'Wake_Steering_2C', 'Wake_Steering_2C_Shift',
           'Robot_Push', 'Robot_Push_Shift', 'Robot_Push_Success', 'Robot_Push_Success_Shift',
           'Plant_Leaf', 'Plant_Leaf_Shift', 'Glucose_Bolus', 'Glucose_Bolus_Shift',
           'Glucose_Bolus_2Meals', 'Glucose_Bolus_2Meals_Shift',
           'Building_Heating', 'Building_Heating_Shift', 'Building_Heating_2C', 'Building_Heating_2C_Shift',
           'Crop_Management', 'Crop_Management_Shift', 'Crop_Management_2C', 'Crop_Management_2C_Shift']

# ---------------------------------------------------------------------------------------------
# max_x E_{P_eval} f(x, c) and min_x E_{P_eval} f(x, c) for the default configurations
# (natural units). Recompute with  python Benchmark/real_world_tools/compute_max_stochastic.py --problem <Name>
# if you change any constructor argument; non-default instances compute them at construction.
# ---------------------------------------------------------------------------------------------
PRECOMPUTED = {
    'Building_Heating_1C[w2;11-12-1-2-3]': (-1.5632107391924859, -14.270797110759675),
    'Building_Heating_1C[w2;3to1-2]': (-1.7028071018621322, -14.680580549114309),
    'Building_Heating_2C[w2;11-12-1-2-3]': (-1.5636289618203398, -14.181567340949753),
    'Building_Heating_2C[w2;3to1-2]': (-1.7247420813449474, -14.830263493455718),
    'Crop_Management_1C[r1;0.5,2,40]': (6121.82957243667, 2923.5465177823585),
    'Crop_Management_1C[r1tor0.5;0.5,2,40]': (6155.526590705542, 1507.6029063907918),
    'Crop_Management_2C[r1T+0;0.5,2,40]': (6058.932012879579, 2915.9791372731224),
    'Crop_Management_2C[r1T+0tor0.75T+1.5;0.5,2,40]': (5222.2602760918635, 1309.999481231972),
    'Glucose_Bolus[adult#001;lunch->breakfast]': (-1.6290493019770738, -15.096064909709131),
    'Glucose_Bolus[adult#001;main_meals->main_meals]': (-2.5287999547261153, -13.564244022664791),
    'Glucose_Bolus_2Meals[adult#001;breakfast_lunch->N60-15_N95-20]': (-1.972983, -44.663777),
    'Glucose_Bolus_2Meals[adult#001;breakfast_lunch->breakfast_lunch]': (-1.173967, -77.413635),
    'Plant_Leaf[0.08,0.06->0.2,0.1]': (128.80261604289495, 56.79169160855065),
    'Plant_Leaf[0.12,0.06->0.12,0.06]': (131.36017462967652, 83.89306096939231),
    'Robot_Push[distance,0.3-0.7,0.6-0.4]': (2.719996, -7.429543),
    'Robot_Push[distance,0.6-0.4,0.6-0.4]': (2.719996, -7.429543),
    'Robot_Push[success,0.3-0.7,0.6-0.4]': (0.404862, 0.0),
    'Robot_Push[success,0.6-0.4,0.6-0.4]': (0.404862, 0.0),
    'Wake_Steering_1C[wd 270,3->268,4]': (0.269437, -0.152474),
    'Wake_Steering_1C[wd 270,3->270,3]': (0.258403, -0.001313),
    'Wake_Steering_2C[wd 270,3->268,4; ws all->sector]': (0.293061, -0.106619),
    'Wake_Steering_2C[wd 270,3->270,3; ws sector->sector]': (0.292245, -0.001396),
}


class _RealWorldProblem(SyntheticTestFunction):
    _check_grad_at_opt = False

    def __init__(self, dim, contexts_dim, negate=False, noise_std=None, bounds=None):
        self.dim = dim
        self.contexts_dim = contexts_dim
        self._bounds = [(0.0, 1.0) for _ in range(dim)]
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.negate_ = negate
        self.can_calculate_stochastic = True

    # ------------------------------------------------------------------ helpers
    @property
    def decision_dim(self):
        return self.dim - self.contexts_dim

    def _sign(self):
        return 1.0 if self.negate_ else -1.0

    def _config_key(self):
        """Identifies the default configuration; None for non-default arguments."""
        return type(self).__name__ if self._is_default else None

    def _init_extrema(self, E_max=None, E_min=None, verbose=True):
        key = self._config_key()
        if E_max is None and key in PRECOMPUTED:
            E_max, E_min = PRECOMPUTED[key]
        if E_max is None:
            if verbose:
                print(f'[{type(self).__name__}] max_stochastic not precomputed for this configuration: '
                      f'computing max_x E f(x, c) numerically ...')
            from_opt = self.optimise_expectation('eval')
            E_max, E_min = from_opt['max'], from_opt['min']
        self.E_max, self.E_min = float(E_max), float(E_min)
        # regret of the *returned* objective: max of +f is E_max, max of -f is -E_min
        self.max_stochastic = self.E_max if self.negate_ else -self.E_min

    # ------------------------------------------------------------------ to implement
    def f_xc(self, X, C):
        """Natural objective on all pairs: X (n_x, d_x), C (n_c, d_c) in [0,1] -> (n_x, n_c)."""
        raise NotImplementedError

    def f_paired(self, X, C):
        """Natural objective on pairs (x_i, c_i): X (n, d_x), C (n, d_c) -> (n,)."""
        return np.array([self.f_xc(X[i:i + 1], C[i:i + 1])[0, 0] for i in range(len(X))])

    def sample_contexts(self, n, which='nominal'):
        raise NotImplementedError

    def context_distribution(self, which='eval'):
        raise NotImplementedError

    # ------------------------------------------------------------------ test-problem interface (main.py)
    def evaluate_true(self, X):
        Xn = X.detach().cpu().double().numpy().reshape(-1, self.decision_dim)
        C = self.sample_contexts(Xn.shape[0], 'nominal')
        f = self.f_paired(Xn, C)
        f_X = torch.as_tensor(self._sign() * f, dtype=X.dtype, device=X.device).reshape(-1)
        contexts = torch.as_tensor(C, dtype=X.dtype, device=X.device).reshape(-1, self.contexts_dim)
        return f_X, contexts

    def evaluate_stochastic(self, X, sample_size=None):
        x = X.detach().cpu().double().numpy().reshape(1, -1)[:, :self.decision_dim]
        val = self.expected_value(x, 'eval')[0]
        return torch.tensor(self._sign() * val, dtype=torch.float64, device=X.device)

    def expected_value(self, X, which='eval'):
        C, w = self.context_distribution(which)
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.f_xc(X, C) @ w

    # ------------------------------------------------------------------ utilities
    def optimise_expectation(self, which='eval', n_sobol=512, n_local=4, seed=0, maxiter=200,
                             verbose=False):
        """Global (Sobol screening) + local (bounded Nelder-Mead) search for max and min of E f."""
        from scipy.optimize import minimize
        d = self.decision_dim
        S = SobolEngine(d, scramble=True, seed=seed).draw(n_sobol).double().numpy()
        S = np.vstack([S, np.full((1, d), 0.5)])
        vals = np.concatenate([self.expected_value(S[i:i + 64], which) for i in range(0, len(S), 64)])
        out = {}
        for name, sgn in (('max', 1.0), ('min', -1.0)):
            order = np.argsort(-sgn * vals)[:n_local]
            best_v, best_x = sgn * vals[order[0]], S[order[0]]
            for i in order:
                res = minimize(lambda z: -sgn * self.expected_value(np.clip(z, 0, 1)[None], which)[0],
                               S[i], method='Nelder-Mead', bounds=[(0, 1)] * d,
                               options={'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-9})
                if -res.fun * 1.0 > best_v:
                    best_v, best_x = -res.fun, np.clip(res.x, 0, 1)
            out[name] = sgn * best_v
            out['argmax' if sgn > 0 else 'argmin'] = best_x
            if verbose:
                print(f'  {name}: {sgn * best_v:.6f} at x = {np.round(best_x, 4)}')
        return out


def _gauss_legendre_01(k):
    z, w = np.polynomial.legendre.leggauss(k)
    return (z + 1) / 2, w / 2  # nodes / weights on [0, 1], weights sum to 1


# =============================================================================================
# 1) Wind-power day-ahead commitment with reserve procurement  (d_x = 2, d_c = 1, real data)
# =============================================================================================
class Wind_Commitment(_RealWorldProblem):
    r"""Day-ahead commitment of a wind farm with reserve procurement.

    Context  c in [0, 1]: realised hourly capacity factor (real data, default: 2010 hourly
             80-m wind speed of a German site shipped with windpowerlib, converted with the
             smoothed Enercon E-82/2300 farm power curve, see Benchmark/scripts/make_wind_cf_windpowerlib.py;
             or any CSV with columns `cf`, `month`, e.g. OPSD country profiles from
             Benchmark/scripts/download_opsd_wind.py).
    Decision x = (x1, x2) in [0, 1]^2: committed energy q = x1 (fraction of capacity) and
             reserve capacity R = max_reserve * x2 bought ahead of time.
    Profit (per MW installed and hour, in units of the day-ahead price):
        f = p q + s (c - q)^+ - r R - a min((q - c)^+, R) - k ((q - c)^+ - R)^+
    i.e. surplus sold at s, shortfall covered by activated reserve at a, rest penalised at k.
    With p=1, s=0.2, r=0.2, a=1.5, k=4 the optimum is the pair of quantiles
        F(q*) = (p - s - r) / (a - s) = 0.4615,   F(q* - R*) = r / (k - a) = 0.08.
    """

    def __init__(self, noise_std=None, negate=False, bounds=None, data_file=None, eval_data_file=None,
                 nominal_months=None, eval_months=None, price=1.0, surplus_price=0.2,
                 reserve_price=0.2, activation_price=1.5, penalty=4.0, max_reserve=0.5,
                 E_max=None, E_min=None, tag=None):
        """data_file: CSV with a `cf` column (and optionally `month`) for the nominal contexts;
        eval_data_file: optional second CSV for the evaluation distribution (e.g. another year);
        nominal_months / eval_months: optional month filters (1..12) applied to the two files."""
        super().__init__(dim=3, contexts_dim=1, negate=negate, noise_std=noise_std, bounds=bounds)
        import pandas as pd
        default_file = os.path.join(DATA_DIR, 'wind_cf_windpowerlib_2010.csv')

        def load(path, months_sel):
            df = pd.read_csv(path)
            cf = np.clip(df['cf'].to_numpy(dtype=float), 0.0, 1.0)
            ok = np.isfinite(cf)
            if months_sel is not None:
                if 'month' not in df:
                    raise ValueError(f'{path} has no `month` column, cannot filter months')
                ok &= np.isin(df['month'].to_numpy(), months_sel)
            return np.sort(cf[ok])

        self.cf_nominal = load(data_file or default_file, nominal_months)
        self.cf_eval = load(eval_data_file or data_file or default_file, eval_months)
        self._cs = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in
                    (('nominal', self.cf_nominal), ('eval', self.cf_eval))}
        self.p, self.s, self.r, self.a, self.k = price, surplus_price, reserve_price, activation_price, penalty
        self.max_reserve = max_reserve
        stem = lambda p: os.path.splitext(os.path.basename(p))[0]
        self.mu = tag or ('windpowerlib2010' if data_file is None else stem(data_file))
        if eval_data_file is not None and tag is None:
            self.mu += 'to' + stem(eval_data_file)
        self.sigma = 'all' if (nominal_months is None and eval_months is None) else \
            f"{'-'.join(map(str, nominal_months or ['all']))}to{'-'.join(map(str, eval_months or ['all']))}"
        self._is_default = False  # optimum is computed exactly at construction (< 1 s)
        self._init_extrema(E_max, E_min, verbose=False)

    # closed-form expectations with the empirical CDF (exact)
    def _E_pos(self, t, which, upper):
        """E (c - t)^+ if upper else E (t - c)^+, for an array t."""
        c, cs = (self.cf_nominal, self._cs['nominal']) if which == 'nominal' else (self.cf_eval, self._cs['eval'])
        n = len(c)
        j = np.searchsorted(c, t, side='right')  # number of atoms <= t
        below = t * j - cs[j]
        if not upper:
            return below / n
        above = (cs[-1] - cs[j]) - t * (n - j)
        return above / n

    def _decode(self, X):
        X = np.atleast_2d(X)
        return X[:, 0], self.max_reserve * X[:, 1]

    def _profit(self, q, R, c):
        short = np.maximum(q - c, 0.0)
        return (self.p * q + self.s * np.maximum(c - q, 0.0) - self.r * R
                - self.a * np.minimum(short, R) - self.k * np.maximum(short - R, 0.0))

    def f_xc(self, X, C):
        q, R = self._decode(X)
        c = np.asarray(C, dtype=float).reshape(1, -1)
        return self._profit(q[:, None], R[:, None], c)

    def f_paired(self, X, C):
        q, R = self._decode(X)
        return self._profit(q, R, np.asarray(C, dtype=float).reshape(-1))

    def expected_value(self, X, which='eval'):
        q, R = self._decode(np.asarray(X, dtype=float))
        e_sur = self._E_pos(q, which, upper=True)
        e_short = self._E_pos(q, which, upper=False)
        e_unc = self._E_pos(q - R, which, upper=False)  # E (q - R - c)^+
        return self.p * q + self.s * e_sur - self.r * R - self.a * (e_short - e_unc) - self.k * e_unc

    def sample_contexts(self, n, which='nominal'):
        c = self.cf_nominal if which == 'nominal' else self.cf_eval
        idx = torch.randint(len(c), (n,)).numpy()
        return c[idx].reshape(-1, 1)

    def context_distribution(self, which='eval'):
        c = self.cf_nominal if which == 'nominal' else self.cf_eval
        return c.reshape(-1, 1), np.full(len(c), 1.0 / len(c))

    def optimise_expectation(self, which='eval', **kw):
        # dense grid with the exact closed form + quantile (first-order) candidate
        g = np.linspace(0, 1, 1001)
        Q, Rn = np.meshgrid(g, g, indexing='ij')
        X = np.stack([Q.ravel(), Rn.ravel()], 1)
        v = np.concatenate([self.expected_value(X[i:i + 200000], which) for i in range(0, len(X), 200000)])
        c = self.cf_nominal if which == 'nominal' else self.cf_eval
        q_star = np.quantile(c, np.clip((self.p - self.s - self.r) / (self.a - self.s), 0, 1))
        lo_star = np.quantile(c, np.clip(self.r / (self.k - self.a), 0, 1))
        cand = np.array([[q_star, np.clip((q_star - lo_star) / self.max_reserve, 0, 1)]])
        vc = self.expected_value(cand, which)
        if vc[0] >= v.max():
            amax, vmax = cand[0], vc[0]
        else:
            amax, vmax = X[v.argmax()], v.max()
        return {'max': float(vmax), 'argmax': amax, 'min': float(v.min()), 'argmin': X[v.argmin()]}


class Wind_Commitment_Shift(Wind_Commitment):
    """Seasonal shift: contexts observed in the windiest months of the year (Nov, Dec, Mar; mean CF
    0.28), regret evaluated in the calmest ones (Jun, Jul; mean CF 0.19)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, nominal_months=(11, 12, 3),
                 eval_months=(6, 7), **kw):
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds,
                         nominal_months=list(nominal_months), eval_months=list(eval_months), **kw)


# =============================================================================================
# 2) Robust wake steering with FLORIS  (d_x = n_turbines - 1, d_c = 1 or 2)
# =============================================================================================
class _WakeSteeringBase(_RealWorldProblem):
    r"""Robust yaw-offset (wake-steering) set-points for a row of NREL-5MW turbines, simulated
    with FLORIS (Gauss-curl-hybrid wake model, NREL, BSD-3 licence).

    Layout: ``n_turbines`` turbines on a west-east line, ``spacing`` rotor diameters apart.
    Decision: yaw offsets of the first n_turbines - 1 turbines in [-yaw_max, yaw_max] deg, i.e.
    the entry of the yaw look-up table for the *measured* wind-direction bin 270 deg (full
    wake overlap). The last turbine has nothing to steer.
    Context: the *true* wind direction in [255, 285] deg, which differs from the measured one
    because of direction variability and vane misalignment; distribution: truncated normal
    N(270 + bias, sd^2) on [255, 285] (FLORIS' UncertainFlorisModel uses the same Gaussian
    model with sd = 3 deg by default). In the 2-context version also the wind speed in
    [3.5, 14.5] m/s, independent of the direction error, distributed as in the real site wind
    rose shipped with FLORIS (examples/inputs/wind_rose.csv; 1 m/s bins, uniform within a bin):
    'sector' = speeds observed with directions in 257.5-282.5 deg, 'all' = all directions.
    The 1-context version uses ws = 8 m/s.

    Objective (objective='gain', default): farm power gain from yawing, in MW,
        f(x, c) = P_farm(yaw(x), c) - P_farm(0, c),
    which has the same expected-value maximiser and stochastic regret as the farm power
    itself (the baseline does not depend on x); objective='power' returns P_farm in MW.
    The optimal yaw changes sign with the sign of the direction bias and shrinks with sd,
    so a mis-specified direction model makes the farm steer the wakes the wrong way.
    """
    WD_SET, WD_HALF = 270.0, 15.0
    WS_CENTRES = np.arange(4.0, 15.0, 1.0)
    WS_LO, WS_HI = 3.5, 14.5
    SECTOR = (257.5, 282.5)

    def __init__(self, contexts_dim, negate=False, noise_std=None, bounds=None,
                 wd_nominal=(270.0, 3.0), wd_eval=(270.0, 3.0), ws_nominal='sector', ws_eval='sector',
                 n_turbines=3, spacing=6.0, yaw_max=25.0, ti=0.06, ws_fixed=8.0, objective='gain',
                 n_wd_nodes=None, n_ws_nodes=6, E_max=None, E_min=None):
        super().__init__(dim=(n_turbines - 1) + contexts_dim, contexts_dim=contexts_dim,
                         negate=negate, noise_std=noise_std, bounds=bounds)
        self.WD_LO, self.WD_HI = self.WD_SET - self.WD_HALF, self.WD_SET + self.WD_HALF
        self.n_turbines, self.spacing, self.yaw_max = n_turbines, spacing, yaw_max
        self.ti, self.ws_fixed, self.objective = ti, ws_fixed, objective
        self.wd_params = {'nominal': tuple(map(float, wd_nominal)), 'eval': tuple(map(float, wd_eval))}
        self.ws_kind = {'nominal': ws_nominal, 'eval': ws_eval}
        self.n_wd_nodes = n_wd_nodes or (64 if contexts_dim == 1 else 32)
        self.n_ws_nodes = n_ws_nodes
        self._fm = self._build_floris()
        self._load_rose()
        self._quad = {}
        fmt = lambda p: f'N{p[0]:g}-{p[1]:g}'
        same = self.wd_params['nominal'] == self.wd_params['eval']
        self.mu = 'ws8' if contexts_dim == 1 else 'ws' + (ws_nominal if ws_nominal == ws_eval else f'{ws_nominal}to{ws_eval}')
        self.sigma = fmt(self.wd_params['nominal']) + ('' if same else 'to' + fmt(self.wd_params['eval']))
        self._is_default = (n_turbines, spacing, yaw_max, ti, ws_fixed, objective, self.n_wd_nodes, n_ws_nodes) == \
            (3, 6.0, 25.0, 0.06, 8.0, 'gain', 64 if contexts_dim == 1 else 32, 6)
        self._init_extrema(E_max, E_min)

    def _config_key(self):
        if not self._is_default:
            return None
        f = lambda p: f'{p[0]:g},{p[1]:g}'
        key = f'Wake_Steering_{self.contexts_dim}C[wd {f(self.wd_params["nominal"])}->{f(self.wd_params["eval"])}'
        if self.contexts_dim == 2:
            key += f'; ws {self.ws_kind["nominal"]}->{self.ws_kind["eval"]}'
        return key + ']'

    # ------------------------------------------------------------------ FLORIS
    def _build_floris(self):
        import logging
        import yaml
        import floris
        from floris import FlorisModel
        logging.getLogger('floris').setLevel(logging.ERROR)
        cfg = FlorisModel.get_defaults()
        lib = os.path.join(os.path.dirname(floris.__file__), 'turbine_library', 'nrel_5MW.yaml')
        with open(lib) as fh:
            turbine = yaml.safe_load(fh)  # passed as a dict: avoids turbine-library path issues
        D = float(turbine['rotor_diameter'])
        cfg['farm'] = {'layout_x': [i * self.spacing * D for i in range(self.n_turbines)],
                       'layout_y': [0.0] * self.n_turbines, 'turbine_type': [turbine]}
        cfg['flow_field'].update(wind_directions=[270.0], wind_speeds=[8.0], turbulence_intensities=[self.ti])
        cfg['wake']['enable_secondary_steering'] = True
        cfg['wake']['enable_yaw_added_recovery'] = True
        cfg['wake']['enable_transverse_velocities'] = True
        cfg['logging'] = {'console': {'enable': False, 'level': 'WARNING'},
                          'file': {'enable': False, 'level': 'WARNING'}}
        return FlorisModel(cfg)

    def _farm_power_MW(self, yaw_deg, wd, ws):
        n = len(wd)
        yaw = np.zeros((n, self.n_turbines))
        yaw[:, :self.n_turbines - 1] = yaw_deg
        out = np.empty(n)
        chunk = 20000
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            m = len(wd[sl])
            self._fm.set(wind_directions=np.asarray(wd[sl], float), wind_speeds=np.asarray(ws[sl], float),
                         turbulence_intensities=np.full(m, self.ti), yaw_angles=yaw[sl])
            self._fm.run()
            out[sl] = self._fm.get_farm_power() / 1e6
        return out

    # ------------------------------------------------------------------ context distributions
    def _load_rose(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(DATA_DIR, 'floris_wind_rose.csv'))
        df = df[df.ws.isin(self.WS_CENTRES)]
        sec = df[(df.wd >= self.SECTOR[0]) & (df.wd <= self.SECTOR[1])]
        p_all = df.groupby('ws')['freq_val'].sum().reindex(self.WS_CENTRES).fillna(0).to_numpy()
        p_sec = sec.groupby('ws')['freq_val'].sum().reindex(self.WS_CENTRES).fillna(0).to_numpy()
        self.ws_tables = {'all': p_all / p_all.sum(), 'sector': p_sec / p_sec.sum()}

    def _wd_trunc(self, which):
        from scipy.stats import norm
        m, s = self.wd_params[which]
        a, b = norm.cdf((self.WD_LO - m) / s), norm.cdf((self.WD_HI - m) / s)
        return m, s, a, b

    def _sample_wd01(self, n, which):
        from scipy.stats import norm
        m, s, a, b = self._wd_trunc(which)
        u = torch.rand(n, dtype=torch.float64).numpy()
        wd = m + s * norm.ppf(a + u * (b - a))
        return (wd - self.WD_LO) / (self.WD_HI - self.WD_LO)

    def _sample_ws01(self, n, which):
        P = self.ws_tables[self.ws_kind[which]]
        cell = torch.multinomial(torch.as_tensor(P, dtype=torch.float64), n, replacement=True).numpy()
        u = torch.rand(n, dtype=torch.float64).numpy()
        return (cell + u) / len(P)

    def sample_contexts(self, n, which='nominal'):
        cols = [self._sample_wd01(n, which)]
        if self.contexts_dim == 2:
            cols.append(self._sample_ws01(n, which))
        return np.stack(cols, 1)

    def _wd_rule(self, which):
        from scipy.stats import norm
        m, s, _, _ = self._wd_trunc(which)
        z, w = _gauss_legendre_01(self.n_wd_nodes)
        wd = self.WD_LO + (self.WD_HI - self.WD_LO) * z
        w = w * norm.pdf((wd - m) / s)
        return z, w / w.sum()

    def _ws_rule(self, which):
        P = self.ws_tables[self.ws_kind[which]]
        z, w = _gauss_legendre_01(self.n_ws_nodes)
        c = ((np.arange(len(P))[:, None] + z[None, :]) / len(P)).ravel()
        ww = (P[:, None] * w[None, :]).ravel()
        keep = ww > 0
        return c[keep], ww[keep] / ww[keep].sum()

    def context_distribution(self, which='eval'):
        """Gauss-Legendre rule: truncated-normal weights for the direction, per-bin rule for the speed."""
        if which in self._quad:
            return self._quad[which]
        zd, wd = self._wd_rule(which)
        if self.contexts_dim == 1:
            C, W = zd.reshape(-1, 1), wd
        else:
            zs, ws = self._ws_rule(which)
            C = np.stack([np.repeat(zd, len(zs)), np.tile(zs, len(zd))], 1)
            W = np.outer(wd, ws).ravel()
        self._quad[which] = (C, W)
        return C, W

    def _decode_contexts(self, C):
        C = np.asarray(C, dtype=float).reshape(-1, self.contexts_dim)
        wd = self.WD_LO + (self.WD_HI - self.WD_LO) * C[:, 0]
        ws = (self.WS_LO + (self.WS_HI - self.WS_LO) * C[:, 1]) if self.contexts_dim == 2 \
            else np.full(len(C), self.ws_fixed)
        return wd, ws

    # ------------------------------------------------------------------ objective
    def _yaw(self, X):
        return self.yaw_max * (2.0 * np.atleast_2d(np.asarray(X, float)) - 1.0)

    def f_paired(self, X, C):
        yaw = self._yaw(X)
        wd, ws = self._decode_contexts(C)
        n = len(wd)
        if self.objective == 'power':
            return self._farm_power_MW(yaw, wd, ws)
        P = self._farm_power_MW(np.vstack([yaw, np.zeros_like(yaw)]), np.concatenate([wd, wd]),
                                np.concatenate([ws, ws]))
        return P[:n] - P[n:]

    def _baseline(self, C):
        key = C.tobytes()
        cache = self.__dict__.setdefault('_base_cache', {})
        if key not in cache:
            if len(cache) > 8:
                cache.clear()
            wd, ws = self._decode_contexts(C)
            cache[key] = self._farm_power_MW(np.zeros((len(wd), self.n_turbines - 1)), wd, ws)
        return cache[key]

    def f_xc(self, X, C):
        yaw = self._yaw(X)
        C = np.asarray(C, dtype=float).reshape(-1, self.contexts_dim)
        wd, ws = self._decode_contexts(C)
        nx, nc = len(yaw), len(wd)
        P = self._farm_power_MW(np.repeat(yaw, nc, axis=0), np.tile(wd, nx), np.tile(ws, nx)).reshape(nx, nc)
        if self.objective == 'power':
            return P
        return P - self._baseline(C)[None, :]


class Wake_Steering(_WakeSteeringBase):
    """d_c = 1 (true wind direction), ws = 8 m/s, P = truncated N(270, 3^2) on [255, 285] deg."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        super().__init__(contexts_dim=1, negate=negate, noise_std=noise_std, bounds=bounds, **kw)


class Wake_Steering_Shift(Wake_Steering):
    """d_c = 1. Contexts observed from the assumed model N(270, 3^2); regret evaluated under
    N(268, 4^2) (a -2 deg vane misalignment and larger variability than assumed)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        kw.setdefault('wd_nominal', (270.0, 3.0))
        kw.setdefault('wd_eval', (268.0, 4.0))
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kw)


class Wake_Steering_2C(_WakeSteeringBase):
    """d_c = 2: (true wind direction, wind speed); direction N(270, 3^2) truncated, speed from
    the site wind rose for the 257.5-282.5 deg sector (real data)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        super().__init__(contexts_dim=2, negate=negate, noise_std=noise_std, bounds=bounds, **kw)


class Wake_Steering_2C_Shift(Wake_Steering_2C):
    """d_c = 2. Contexts observed from the assumed model (direction N(270, 3^2), speed = site-wide
    wind-rose distribution over all directions); regret evaluated under direction N(268, 4^2)
    and the real speed distribution of the waked sector (lower speeds)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        kw.setdefault('wd_nominal', (270.0, 3.0))
        kw.setdefault('wd_eval', (268.0, 4.0))
        kw.setdefault('ws_nominal', 'all')
        kw.setdefault('ws_eval', 'sector')
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kw)


# =============================================================================================
# 3) Robot pushing with a random goal  (d_x = 3, d_c = 2, Box2D)
# =============================================================================================
class Robot_Push(_RealWorldProblem):
    r"""3-D robot pushing of Wang & Jegelka (2017) with the goal as a continuous context.

    Decision x in [0,1]^3 -> hand start (rx, ry) = 10 x[:2] - 5 in [-5, 5]^2 and push duration
    t = 1 + 29 x[2] in [1, 30] (10 t simulation steps).  The object (disc of radius 1) starts at
    the origin; the hand pushes towards it, the object ends at e(x) (Box2D simulation).
    Context c in [0,1]^2 -> goal g = 10 c - 5 in [-5, 5]^2.
    Reward 'distance' (as in the original benchmark):  f = 5 - ||e(x) - g||;
    reward 'success':                                   f = exp(-||e(x) - g||^2 / 2)
    (with 'distance' the Wasserstein worst case of every x is E_P f - eps, so robustness does not
    change the decision; 'success' is the variant where it does).
    Goal distribution: two-component Gaussian mixture (clamped to the table), default
    0.6 N((2.5, 2.0), 0.7^2 I) + 0.4 N((-2.0, 3.0), 1.2^2 I).  The object's end position does not
    depend on the goal, so E_P f needs one simulation per x (cached).
    """
    MEANS = ((2.5, 2.0), (-2.0, 3.0))
    STDS = (0.7, 1.2)

    def __init__(self, noise_std=None, negate=False, bounds=None, reward='distance',
                 weights_nominal=(0.6, 0.4), weights_eval=(0.6, 0.4), push_noise=0.0,
                 n_qmc=16384, n_support=1024, E_max=None, E_min=None):
        super().__init__(dim=5, contexts_dim=2, negate=negate, noise_std=noise_std, bounds=bounds)
        from .push_sim import push_object_final_position  # needs Box2D
        self._push = push_object_final_position
        self.reward, self.push_noise = reward, push_noise
        self.w = {'nominal': np.asarray(weights_nominal, float), 'eval': np.asarray(weights_eval, float)}
        self.n_qmc, self.n_support = n_qmc, n_support
        self._cache = {}
        self._qmc = {}
        self.mu = f'{reward}'
        self.sigma = 'w' + '-'.join(f'{v:g}' for v in weights_nominal) + \
                     ('' if weights_nominal == weights_eval else 'to' + '-'.join(f'{v:g}' for v in weights_eval))
        self._is_default = (push_noise == 0.0 and n_qmc == 16384)
        self._init_extrema(E_max, E_min)

    def _config_key(self):
        if not self._is_default:
            return None
        fmt = lambda w: '-'.join(f'{float(v):g}' for v in w)
        return f'Robot_Push[{self.reward},{fmt(self.w["nominal"])},{fmt(self.w["eval"])}]'

    # ---------------------------------------------------------------- simulation
    def end_positions(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        out = np.empty((len(X), 2))
        for i, x in enumerate(X):
            key = tuple(np.round(x, 12))
            if key not in self._cache:
                rx, ry, t = 10 * x[0] - 5, 10 * x[1] - 5, 1 + 29 * x[2]
                self._cache[key] = self._push(rx, ry, t, push_noise=0.0)
            out[i] = self._cache[key]
        return out

    def _reward(self, d):
        return 5.0 - d if self.reward == 'distance' else np.exp(-0.5 * d ** 2)

    def f_xc(self, X, C):
        e = self.end_positions(X)
        g = 10.0 * np.asarray(C, float).reshape(-1, 2) - 5.0
        d = np.linalg.norm(e[:, None, :] - g[None, :, :], axis=-1)
        return self._reward(d)

    def f_paired(self, X, C):
        X = np.atleast_2d(np.asarray(X, float))
        g = 10.0 * np.asarray(C, float).reshape(-1, 2) - 5.0
        if self.push_noise > 0:  # noisy observation of the push (expectation uses the noiseless sim)
            rng = np.random.default_rng(int(torch.randint(2 ** 31 - 1, (1,)).item()))
            e = np.stack([self._push(10 * x[0] - 5, 10 * x[1] - 5, 1 + 29 * x[2], self.push_noise, rng)
                          for x in X])
        else:
            e = self.end_positions(X)
        return self._reward(np.linalg.norm(e - g, axis=1))

    # ---------------------------------------------------------------- contexts
    def _mixture_from_uniform(self, U, weights):
        """U (n, 3) uniforms -> clamped mixture samples in normalised [0,1]^2 (inverse transform)."""
        cum = np.cumsum(weights / weights.sum())
        comp = np.searchsorted(cum, U[:, 0], side='right').clip(max=len(weights) - 1)
        z = math.sqrt(2.0) * torch.erfinv(torch.as_tensor(2 * U[:, 1:] - 1).clamp(-1 + 1e-12, 1 - 1e-12)).numpy()
        m = np.asarray(self.MEANS)[comp]
        s = np.asarray(self.STDS)[comp][:, None]
        g = m + s * z
        return np.clip((g + 5.0) / 10.0, 0.0, 1.0)

    def sample_contexts(self, n, which='nominal'):
        U = torch.rand(n, 3, dtype=torch.float64).numpy()
        return self._mixture_from_uniform(U, self.w[which])

    def context_distribution(self, which='eval', n=None):
        n = n or self.n_qmc
        key = (which, n)
        if key not in self._qmc:
            U = SobolEngine(3, scramble=True, seed=1234).draw(n).double().numpy()
            self._qmc[key] = (self._mixture_from_uniform(U, self.w[which]), np.full(n, 1.0 / n))
        return self._qmc[key]

    def optimise_expectation(self, which='eval', n_sobol=4096, n_local=8, **kw):
        return super().optimise_expectation(which, n_sobol=n_sobol, n_local=n_local, **kw)


class Robot_Push_Shift(Robot_Push):
    """Goal-frequency (label) shift: contexts observed with mixture weights (0.3, 0.7),
    regret evaluated with (0.6, 0.4)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        kw.setdefault('weights_nominal', (0.3, 0.7))
        kw.setdefault('weights_eval', (0.6, 0.4))
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kw)


class Robot_Push_Success(Robot_Push):
    """Robot_Push with the success reward exp(-d^2/2)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        kw.setdefault('reward', 'success')
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kw)


class Robot_Push_Success_Shift(Robot_Push_Shift):
    """Robot_Push_Shift with the success reward exp(-d^2/2)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        kw.setdefault('reward', 'success')
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kw)


# =============================================================================================
# Shared helpers for the problems below
# =============================================================================================
def _truncnorm_rule(mean, sd, lo, hi, n):
    """Gauss-Legendre nodes on [0, 1] (normalised [lo, hi]) with truncated-normal weights."""
    from scipy.stats import norm
    z, w = _gauss_legendre_01(n)
    v = lo + (hi - lo) * z
    w = w * norm.pdf((v - mean) / sd)
    return z, w / w.sum()


def _mixture_rule(components, lo, hi, n):
    """components: list of (weight, mean, sd) in physical units; truncated to [lo, hi]."""
    from scipy.stats import norm
    z, w = _gauss_legendre_01(n)
    v = lo + (hi - lo) * z
    p = sum(a * norm.pdf(v, m, s) for a, m, s in components)
    w = w * p
    return z, w / w.sum()


def _sample_mixture01(components, lo, hi, n):
    """Exact samples of a truncated normal mixture, normalised to [0, 1] (torch RNG)."""
    from scipy.stats import norm
    comps = np.asarray(components, float)
    a = norm.cdf((lo - comps[:, 1]) / comps[:, 2])
    b = norm.cdf((hi - comps[:, 1]) / comps[:, 2])
    mass = comps[:, 0] * (b - a)  # truncation changes the component weights
    pick = torch.multinomial(torch.as_tensor(mass / mass.sum(), dtype=torch.float64), n, replacement=True).numpy()
    u = torch.rand(n, dtype=torch.float64).numpy()
    v = comps[pick, 1] + comps[pick, 2] * norm.ppf(a[pick] + u * (b[pick] - a[pick]))
    return (np.clip(v, lo, hi) - lo) / (hi - lo)


def _fmt_mix(components):
    return '+'.join(f'N{m:g}-{s:g}' if len(components) == 1 else f'{a:.2g}N{m:g}-{s:g}' for a, m, s in components)


class _MaternGP:
    """Posterior mean of an exact GP (ScaleKernel(Matern-5/2, ARD), linear mean function) fitted offline;
    see Benchmark/scripts/generate_*_data.py --fit. Inputs in [0, 1]^d."""

    def __init__(self, npz, prefix):
        g = lambda k: npz[f'{prefix}_{k}']
        self.X, self.alpha, self.ls = g('X'), g('alpha'), g('ls')
        self.os, self.ym, self.ys = float(g('os')), float(g('ym')), float(g('ys'))
        self.mw, self.mb = g('mean_w'), float(g('mean_b'))

    def __call__(self, Xq, chunk=4096):
        Xq = np.atleast_2d(np.asarray(Xq, float))
        out = np.empty(len(Xq))
        B = self.X / self.ls
        b2 = (B ** 2).sum(1)
        for i in range(0, len(Xq), chunk):
            A = Xq[i:i + chunk] / self.ls
            r = np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + b2[None, :] - 2.0 * A @ B.T, 0.0))
            s5r = math.sqrt(5.0) * r
            K = self.os * (1.0 + s5r + 5.0 / 3.0 * r ** 2) * np.exp(-s5r)
            out[i:i + chunk] = (Xq[i:i + chunk] @ self.mw + self.mb + K @ self.alpha) * self.ys + self.ym
        return out


class _EmpiricalDaysMixin:
    """Context distribution = empirical distribution of real days (atoms), sampled by resampling."""

    def sample_contexts(self, n, which='nominal'):
        C = self._atoms[which]
        idx = torch.randint(len(C), (n,)).numpy()
        return C[idx]

    def context_distribution(self, which='eval'):
        C = self._atoms[which]
        return C, np.full(len(C), 1.0 / len(C))


# =============================================================================================
# 4) Plant maximum leaf area (GP fitted to real lab data, Tay et al. 2022)  (d_x = 1, d_c = 1)
# =============================================================================================
class Plant_Leaf(_RealWorldProblem):
    r"""Maximum leaf area of plants grown in nutrient solution, as a function of the solution pH
    (decision) and the ammonia concentration NH3 (context).

    The objective is the posterior mean of the heteroscedastic sparse GP of Tay et al. (ICML 2022,
    "Efficient distributionally robust Bayesian optimization with worst-case sensitivity") fitted to
    real leaf-area measurements (Temasek Life Sciences Laboratory); parameters from
    https://github.com/sebtsh/efficient-distributionally-robust-BO (MIT), data/plant/NH3pH_gp_leaf_dict.p,
    converted to numpy (Benchmark/data/plant_leaf_NH3pH_svgp.npz; agreement with gpflow < 1e-11).
    x in [0, 1] -> pH = 2.5 + 4 x;  c in [0, 1] -> NH3 = 30000 c uM (the ranges used by Tay et al.).
    f = maximum leaf area (units of the data set), maximised.
    Context distribution (normalised NH3): truncated normal N(m, s^2) on [0, 1], default
    N(0.12, 0.06^2) -- straddles the change of regime at ~ 0.15 (acidic pH optimal below, pH 5-6
    above), so E f has two near-equal local maxima (pH ~ 3.5 and ~ 5.2).
    """

    def __init__(self, noise_std=None, negate=False, bounds=None, nh3_nominal=(0.12, 0.06),
                 nh3_eval=(0.12, 0.06), n_nodes=64, E_max=None, E_min=None):
        super().__init__(dim=2, contexts_dim=1, negate=negate, noise_std=noise_std, bounds=bounds)
        D = np.load(os.path.join(DATA_DIR, 'plant_leaf_NH3pH_svgp.npz'))
        self._Z, self._alpha = D['Z'], D['alpha']
        self._var, self._ls = float(D['variance']), D['lengthscales']
        self._lm, self._lsd = float(D['leaf_mean']), float(D['leaf_std'])
        self.params = {'nominal': tuple(map(float, nh3_nominal)), 'eval': tuple(map(float, nh3_eval))}
        self.n_nodes = n_nodes
        f = lambda p: f'N{p[0]:g}-{p[1]:g}'
        self.mu = 'NH3'
        self.sigma = f(self.params['nominal']) + ('' if self.params['nominal'] == self.params['eval']
                                                  else 'to' + f(self.params['eval']))
        self._is_default = n_nodes == 64
        self._init_extrema(E_max, E_min)

    def _config_key(self):
        if not self._is_default:
            return None
        f = lambda p: f'{p[0]:g},{p[1]:g}'
        return f'Plant_Leaf[{f(self.params["nominal"])}->{f(self.params["eval"])}]'

    def _leaf(self, pH, nh3):
        X = np.stack([nh3, pH], 1) / self._ls
        B = self._Z / self._ls
        d2 = (X ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * X @ B.T
        return (self._var * np.exp(-0.5 * np.maximum(d2, 0.0))) @ self._alpha * self._lsd + self._lm

    def f_xc(self, X, C):
        X = np.atleast_2d(np.asarray(X, float))
        C = np.asarray(C, float).reshape(-1)
        pH = np.repeat(2.5 + 4.0 * X[:, 0], len(C))
        nh3 = np.tile(30000.0 * C, len(X))
        return self._leaf(pH, nh3).reshape(len(X), len(C))

    def f_paired(self, X, C):
        X = np.atleast_2d(np.asarray(X, float))
        return self._leaf(2.5 + 4.0 * X[:, 0], 30000.0 * np.asarray(C, float).reshape(-1))

    def sample_contexts(self, n, which='nominal'):
        m, s = self.params[which]
        return _sample_mixture01([(1.0, m, s)], 0.0, 1.0, n).reshape(-1, 1)

    def context_distribution(self, which='eval'):
        m, s = self.params[which]
        z, w = _truncnorm_rule(m, s, 0.0, 1.0, self.n_nodes)
        return z.reshape(-1, 1), w


class Plant_Leaf_Shift(Plant_Leaf):
    """NH3 observed from N(0.08, 0.06^2) (acidic pH optimal), regret under N(0.20, 0.10^2) (pH ~5.3 optimal)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        kw.setdefault('nh3_nominal', (0.08, 0.06))
        kw.setdefault('nh3_eval', (0.20, 0.10))
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kw)


# =============================================================================================
# 5) Insulin bolus dosing with the UVA/Padova simulator (simglucose port)  (d_c = 1 or 2)
# =============================================================================================
SIMGLUCOSE_MEALS = {  # meal-size distributions (g CHO) of simglucose's RandomScenario
    'breakfast': [(1.0, 45.0, 10.0)], 'lunch': [(1.0, 70.0, 10.0)], 'dinner': [(1.0, 80.0, 10.0)],
    'main_meals': [(1 / 3, 45.0, 10.0), (1 / 3, 70.0, 10.0), (1 / 3, 80.0, 10.0)]}


class Glucose_Bolus(_RealWorldProblem):
    r"""Pre-meal insulin bolus for a type-1-diabetes patient, in silico (UVA/Padova model as
    implemented in simglucose; vectorised port in Benchmark/glucose_sim.py, agreement < 0.05 mg/dL).

    Patient: virtual patient `adult#001` (any of the 30 simglucose patients can be passed).
    Decision x in [0, 1] -> bolus dose = 25 x U, delivered 15 min before the meal (usual advice).
    Context c in [0, 1] -> carbohydrate content of the meal = 10 + 140 c g (unknown when dosing).
    f = - mean Kovatchev blood-glucose risk index (LBGI + HBGI per minute, as in simglucose) from
    the bolus to 6 h after the meal start (maximised; 0 is perfect, hypoglycaemia is penalised
    much more than hyperglycaemia).
    Context distribution: simglucose's RandomScenario meal sizes, default the mixture of its three
    main meals (breakfast N(45, 10^2), lunch N(70, 10^2), dinner N(80, 10^2), equal weights).
    Note: decisions on the timing or shape of insulin delivery were tried and discarded -- in this
    model they are always optimal at the bound (earliest / front-loaded), because every virtual
    patient's basal glucose is >= 133 mg/dL.
    In-silico benchmark only, not medical advice.
    """
    DOSE_MAX = 25.0
    CHO_LO, CHO_HI = 10.0, 150.0
    T_BOLUS, HORIZON = -15.0, 360.0

    def __init__(self, noise_std=None, negate=False, bounds=None, patient='adult#001',
                 meals_nominal='main_meals', meals_eval='main_meals', n_nodes=48, E_max=None, E_min=None):
        super().__init__(dim=2, contexts_dim=1, negate=negate, noise_std=noise_std, bounds=bounds)
        from .glucose_sim import load_patient, simulate_schedule
        self._P = load_patient(patient)
        self._sim = simulate_schedule
        get = lambda m: SIMGLUCOSE_MEALS[m] if isinstance(m, str) else [tuple(map(float, c)) for c in m]
        self.meals = {'nominal': get(meals_nominal), 'eval': get(meals_eval)}
        self.patient, self.n_nodes = patient, n_nodes
        name = lambda m: m if isinstance(m, str) else _fmt_mix(get(m))
        self.mu = patient.replace('#', '')
        self.sigma = name(meals_nominal) + ('' if meals_nominal == meals_eval else 'to' + name(meals_eval))
        self._key = f'{type(self).__name__.split("_Shift")[0]}[{patient};{name(meals_nominal)}->{name(meals_eval)}]'
        self._is_default = n_nodes == 48
        self._init_extrema(E_max, E_min)

    def _config_key(self):
        return self._key if self._is_default else None

    def _risk(self, dose, cho, chunk=20000):
        out = np.empty(len(dose))
        for i in range(0, len(dose), chunk):
            sl = slice(i, i + chunk)
            out[sl] = self._sim(self._P, [0.0], cho[sl, None], [[self.T_BOLUS]], dose[sl, None],
                                t_start=self.T_BOLUS, t_end=self.HORIZON, mean_risk_from=self.T_BOLUS)
        return out

    def f_paired(self, X, C):
        X = np.atleast_2d(np.asarray(X, float))
        cho = self.CHO_LO + (self.CHO_HI - self.CHO_LO) * np.asarray(C, float).reshape(-1)
        return -self._risk(self.DOSE_MAX * X[:, 0], cho)

    def f_xc(self, X, C):
        X = np.atleast_2d(np.asarray(X, float))
        C = np.asarray(C, float).reshape(-1)
        dose = np.repeat(self.DOSE_MAX * X[:, 0], len(C))
        cho = np.tile(self.CHO_LO + (self.CHO_HI - self.CHO_LO) * C, len(X))
        return -self._risk(dose, cho).reshape(len(X), len(C))

    def sample_contexts(self, n, which='nominal'):
        return _sample_mixture01(self.meals[which], self.CHO_LO, self.CHO_HI, n).reshape(-1, 1)

    def context_distribution(self, which='eval'):
        z, w = _mixture_rule(self.meals[which], self.CHO_LO, self.CHO_HI, self.n_nodes)
        return z.reshape(-1, 1), w

    def optimise_expectation(self, which='eval', **kw):
        # 1-D decision: dense grid + golden-section refinement (exact up to the quadrature)
        from scipy.optimize import minimize_scalar
        g = np.linspace(0, 1, 201)
        v = self.expected_value(g[:, None], which)
        out = {}
        for name, sgn in (('max', 1.0), ('min', -1.0)):
            i = int(np.argmax(sgn * v))
            lo, hi = g[max(i - 1, 0)], g[min(i + 1, len(g) - 1)]
            r = minimize_scalar(lambda z: -sgn * self.expected_value(np.array([[z]]), which)[0], bounds=(lo, hi),
                                method='bounded', options={'xatol': 1e-6})
            best = max((sgn * v[i], g[i]), (-r.fun, r.x))
            out[name], out['argmax' if sgn > 0 else 'argmin'] = sgn * best[0], np.array([best[1]])
        return out


class Glucose_Bolus_Shift(Glucose_Bolus):
    """Bolus learned on lunch-size meals N(70, 10^2), regret on breakfast-size meals N(45, 10^2)
    (over-dosing -> hypoglycaemia risk)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        kw.setdefault('meals_nominal', 'lunch')
        kw.setdefault('meals_eval', 'breakfast')
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kw)


class Glucose_Bolus_2Meals(Glucose_Bolus):
    r"""Two meals: breakfast at t = 0 and lunch at t = 300 min, each with a bolus 15 min before.
    x in [0, 1]^2 -> (breakfast dose, lunch dose) = 25 x U; c in [0, 1]^2 -> (breakfast, lunch)
    carbohydrates = 10 + 140 c g. The doses interact through the insulin on board at lunch.
    f = - mean risk index from the first bolus to 6 h after lunch.
    Default P: simglucose's breakfast N(45, 10^2) x lunch N(70, 10^2), independent."""

    MEAL_TIMES = (0.0, 300.0)

    def __init__(self, noise_std=None, negate=False, bounds=None, patient='adult#001',
                 meals_nominal=('breakfast', 'lunch'), meals_eval=('breakfast', 'lunch'), n_nodes=16,
                 E_max=None, E_min=None):
        _RealWorldProblem.__init__(self, dim=4, contexts_dim=2, negate=negate, noise_std=noise_std, bounds=bounds)
        from .glucose_sim import load_patient, simulate_schedule
        self._P = load_patient(patient)
        self._sim = simulate_schedule
        get = lambda m: SIMGLUCOSE_MEALS[m] if isinstance(m, str) else [tuple(map(float, c)) for c in m]
        self.meals = {'nominal': [get(m) for m in meals_nominal], 'eval': [get(m) for m in meals_eval]}
        self.patient, self.n_nodes = patient, n_nodes
        name = lambda ms: '_'.join(m if isinstance(m, str) else _fmt_mix(get(m)) for m in ms)
        self.mu = patient.replace('#', '')
        self.sigma = name(meals_nominal) + ('' if tuple(meals_nominal) == tuple(meals_eval) else 'to' + name(meals_eval))
        self._key = f'Glucose_Bolus_2Meals[{patient};{name(meals_nominal)}->{name(meals_eval)}]'
        self._is_default = n_nodes == 16
        self._init_extrema(E_max, E_min)

    def _risk2(self, doses, chos, chunk=8000):
        out = np.empty(len(doses))
        bt = np.array([[t + self.T_BOLUS for t in self.MEAL_TIMES]])
        for i in range(0, len(doses), chunk):
            sl = slice(i, i + chunk)
            out[sl] = self._sim(self._P, list(self.MEAL_TIMES), chos[sl], bt, doses[sl], t_start=self.T_BOLUS,
                                t_end=self.MEAL_TIMES[-1] + self.HORIZON, mean_risk_from=self.T_BOLUS)
        return out

    def _cho(self, C):
        return self.CHO_LO + (self.CHO_HI - self.CHO_LO) * np.asarray(C, float).reshape(-1, 2)

    def f_paired(self, X, C):
        X = np.atleast_2d(np.asarray(X, float))
        return -self._risk2(self.DOSE_MAX * X[:, :2], self._cho(C))

    def f_xc(self, X, C):
        X = np.atleast_2d(np.asarray(X, float))
        cho = self._cho(C)
        doses = np.repeat(self.DOSE_MAX * X[:, :2], len(cho), axis=0)
        return -self._risk2(doses, np.tile(cho, (len(X), 1))).reshape(len(X), len(cho))

    def sample_contexts(self, n, which='nominal'):
        return np.stack([_sample_mixture01(m, self.CHO_LO, self.CHO_HI, n) for m in self.meals[which]], 1)

    def context_distribution(self, which='eval'):
        (z1, w1), (z2, w2) = (_mixture_rule(m, self.CHO_LO, self.CHO_HI, self.n_nodes) for m in self.meals[which])
        C = np.stack([np.repeat(z1, len(z2)), np.tile(z2, len(z1))], 1)
        return C, np.outer(w1, w2).ravel()

    def optimise_expectation(self, which='eval', **kw):
        kw.setdefault('n_sobol', 256)
        kw.setdefault('n_local', 2)
        kw.setdefault('maxiter', 120)
        return _RealWorldProblem.optimise_expectation(self, which, **kw)


class Glucose_Bolus_2Meals_Shift(Glucose_Bolus_2Meals):
    """Two meals; contexts from simglucose's breakfast x lunch sizes, regret under larger and more
    variable meals N(60, 15^2) x N(95, 20^2)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        kw.setdefault('meals_eval', ([(1.0, 60.0, 15.0)], [(1.0, 95.0, 20.0)]))
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kw)


# =============================================================================================
# 6) Building heating schedule, BOPTEST bestest_hydronic (GP surrogate of the FMU)  (d_c = 1 or 2)
# =============================================================================================
class Building_Heating(_EmpiricalDaysMixin, _RealWorldProblem):
    r"""Heating-schedule tuning for the BOPTEST test case `bestest_hydronic` (IBPSA Project 1:
    single-zone dwelling, 48 m2, gas boiler + radiator, Brussels weather; occupied before 07:00 and
    after 20:00), surrogate of the Modelica/FMU emulator.

    Decision x in [0, 1]^2 -> occupied heating setpoint T_occ = 20.5 + 2 x1 degC and pre-heating lead
    time = 4 x2 h before the evening occupancy (setback 15 degC in between), same schedule every day.
    Context: daily mean outdoor temperature in [-6, 14] degC (and daily global horizontal irradiation
    in [0, 4] kWh/m2 for the 2-context version), normalised.
    f = -(energy cost + w * thermal discomfort) per day, EUR (maximised): cost with the test-case
    prices (gas 0.0498, electricity 0.2535 EUR/kWh), discomfort in Kh w.r.t. the BOPTEST comfort band
    (21-24 degC occupied, 15-30 degC otherwise), w = 2 EUR/Kh by default.
    Surrogate: posterior means of exact GPs (Matern-5/2 ARD) fitted to 2500 FMU simulations
    (Sobol design over x and the real days; Benchmark/scripts/generate_boptest_data.py), one for the cost and
    one for the discomfort (both clipped at 0).
    Context distribution: empirical distribution of the 142 real days Nov-Mar of the test-case weather
    year (default) -- the shift variants use March (mild) as nominal and January-February (cold) as eval:
    the optimal pre-heating grows from ~ 1.15 h to ~ 1.8 h.
    """
    T_LO, T_HI, H_LO, H_HI = -6.0, 14.0, 0.0, 4.0

    def __init__(self, noise_std=None, negate=False, bounds=None, contexts_dim=1, months_nominal=(11, 12, 1, 2, 3),
                 months_eval=(11, 12, 1, 2, 3), discomfort_weight=2.0, E_max=None, E_min=None):
        _RealWorldProblem.__init__(self, dim=2 + contexts_dim, contexts_dim=contexts_dim, negate=negate,
                                   noise_std=noise_std, bounds=bounds)
        import pandas as pd
        npz = np.load(os.path.join(DATA_DIR, 'boptest_hydronic_gp.npz'))
        tag = f'{contexts_dim}C'
        self._gp_cost, self._gp_tdis = _MaternGP(npz, f'{tag}_cost'), _MaternGP(npz, f'{tag}_tdis')
        days = pd.read_csv(os.path.join(DATA_DIR, 'boptest_hydronic_days.csv'))
        cols = [(days.Tout.to_numpy() - self.T_LO) / (self.T_HI - self.T_LO)]
        if contexts_dim == 2:
            cols.append((days.Hglo.to_numpy() - self.H_LO) / (self.H_HI - self.H_LO))
        C = np.clip(np.stack(cols, 1), 0.0, 1.0)
        self._atoms = {'nominal': C[days.month.isin(months_nominal).to_numpy()],
                       'eval': C[days.month.isin(months_eval).to_numpy()]}
        self.w_dis = discomfort_weight
        self.mu = f'w{discomfort_weight:g}'
        mfmt = lambda m: '-'.join(map(str, m))
        self.sigma = mfmt(months_nominal) + ('' if tuple(months_nominal) == tuple(months_eval) else 'to' + mfmt(months_eval))
        self._key = f'Building_Heating_{tag}[w{discomfort_weight:g};{self.sigma}]'
        self._is_default = True
        self._init_extrema(E_max, E_min)

    def _config_key(self):
        return self._key

    def _f(self, Z):
        return -(np.maximum(self._gp_cost(Z), 0.0) + self.w_dis * np.maximum(self._gp_tdis(Z), 0.0))

    def f_paired(self, X, C):
        X = np.atleast_2d(np.asarray(X, float))[:, :2]
        return self._f(np.hstack([X, np.asarray(C, float).reshape(len(X), -1)]))

    def f_xc(self, X, C):
        X = np.atleast_2d(np.asarray(X, float))[:, :2]
        C = np.asarray(C, float).reshape(-1, self.contexts_dim)
        Z = np.hstack([np.repeat(X, len(C), axis=0), np.tile(C, (len(X), 1))])
        return self._f(Z).reshape(len(X), len(C))


class Building_Heating_Shift(Building_Heating):
    """Schedule learned on the mild March days, regret on the cold January-February days."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        kw.setdefault('months_nominal', (3,))
        kw.setdefault('months_eval', (1, 2))
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kw)


class Building_Heating_2C(Building_Heating):
    """Contexts (daily mean outdoor temperature, daily global horizontal irradiation)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, contexts_dim=2, **kw)


class Building_Heating_2C_Shift(Building_Heating_Shift):
    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, contexts_dim=2, **kw)


# =============================================================================================
# 7) Potato crop management with WOFOST 8.1 / PCSE (GP surrogate)  (d_x = 2, d_c = 1 or 2)
# =============================================================================================
class Crop_Management(_RealWorldProblem):
    r"""Nitrogen fertiliser and irrigation for a potato crop (Fontane, planted 20 April, harvested
    20 September) on a sandy soil (EC1) at Wageningen, simulated with WOFOST 8.1 (PCSE,
    Wofost81_NWLP_CWB_CNB: water- and nitrogen-limited), surrogate of the crop model.

    Decision x in [0, 1]^2 -> total mineral N = 250 x1 kg/ha (60 % at planting, 40 % 40 days later)
    and total irrigation = 40 x2 cm (10 weekly doses from 1 June).
    Context: rainfall multiplier rho in [0.2, 1.8] applied to the real 2006 Wageningen weather (and a
    temperature offset dT in [-2, 4] degC for the 2-context version; evapotranspiration recomputed).
    f = profit in EUR/ha = 0.5 EUR/kg x tuber dry matter (~105 EUR/t fresh) - 2 EUR/kg N x N
        - 40 EUR/cm x irrigation (4 EUR/mm, maximised). Optimum ~ 115 kg N/ha and ~ 26 cm irrigation;
        the N response is concave up to ~ 100 kg/ha and nearly linear beyond, so a lower N price
        (< 1.5 EUR/kg) moves the optimal N to the bound.
    Surrogate: posterior mean of an exact GP (Matern-5/2 ARD) fitted to 2500 WOFOST runs (Sobol design;
    Benchmark/scripts/generate_wofost_data.py), in-sample RMSE reported by the script.
    Context distribution: rho ~ log-normal fitted to the real inter-annual variability of the
    planting-to-harvest rainfall at Wageningen (2004-2008, 2013-2015, relative to 2006:
    log-mean -0.024, log-sd 0.164) and dT ~ N(0, 0.59^2) (sd of the seasonal mean temperature over
    the same years), truncated to the boxes. Shift variants: severe drought (rain x 0.5) for 1 context;
    drier and warmer (rain x 0.75, +1.5 degC) for 2 contexts.
    """
    N_MAX, I_MAX = 250.0, 40.0
    RHO_LO, RHO_HI, DT_LO, DT_HI = 0.2, 1.8, -2.0, 4.0
    LOG_RHO = (-0.0244, 0.1641)
    DT_SD = 0.59

    def __init__(self, noise_std=None, negate=False, bounds=None, contexts_dim=1, rain_factor_eval=1.0,
                 dT_eval=0.0, rain_factor_nominal=1.0, dT_nominal=0.0, price=0.5, price_N=2.0, price_irr=40.0,
                 n_nodes=None, E_max=None, E_min=None):
        _RealWorldProblem.__init__(self, dim=2 + contexts_dim, contexts_dim=contexts_dim, negate=negate,
                                   noise_std=noise_std, bounds=bounds)
        npz = np.load(os.path.join(DATA_DIR, 'wofost_potato_gp.npz'))
        self._gp = _MaternGP(npz, f'{contexts_dim}C')
        self.scen = {'nominal': (rain_factor_nominal, dT_nominal), 'eval': (rain_factor_eval, dT_eval)}
        self.price, self.price_N, self.price_irr = price, price_N, price_irr
        self.n_nodes = n_nodes or ((64,) if contexts_dim == 1 else (32, 16))
        f = lambda s: f'r{s[0]:g}' + (f'T{s[1]:+g}' if contexts_dim == 2 else '')
        self.mu = 'potatoNL'
        self.sigma = f(self.scen['nominal']) + ('' if self.scen['nominal'] == self.scen['eval'] else 'to' + f(self.scen['eval']))
        self._key = f'Crop_Management_{contexts_dim}C[{self.sigma};{price:g},{price_N:g},{price_irr:g}]'
        self._is_default = n_nodes is None
        self._init_extrema(E_max, E_min)

    def _config_key(self):
        return self._key if self._is_default else None

    def _profit(self, Z):
        twso = np.maximum(self._gp(Z), 0.0)
        return self.price * twso - self.price_N * self.N_MAX * Z[:, 0] - self.price_irr * self.I_MAX * Z[:, 1]

    def f_paired(self, X, C):
        X = np.atleast_2d(np.asarray(X, float))[:, :2]
        return self._profit(np.hstack([X, np.asarray(C, float).reshape(len(X), -1)]))

    def f_xc(self, X, C):
        X = np.atleast_2d(np.asarray(X, float))[:, :2]
        C = np.asarray(C, float).reshape(-1, self.contexts_dim)
        Z = np.hstack([np.repeat(X, len(C), axis=0), np.tile(C, (len(X), 1))])
        return self._profit(Z).reshape(len(X), len(C))

    def _rho_params(self, which):
        return self.LOG_RHO[0] + math.log(self.scen[which][0]), self.LOG_RHO[1]

    def sample_contexts(self, n, which='nominal'):
        from scipy.stats import norm
        m, s = self._rho_params(which)
        a, b = norm.cdf((math.log(self.RHO_LO) - m) / s), norm.cdf((math.log(self.RHO_HI) - m) / s)
        u = torch.rand(n, dtype=torch.float64).numpy()
        rho = np.exp(m + s * norm.ppf(a + u * (b - a)))
        cols = [(rho - self.RHO_LO) / (self.RHO_HI - self.RHO_LO)]
        if self.contexts_dim == 2:
            cols.append(_sample_mixture01([(1.0, self.scen[which][1], self.DT_SD)], self.DT_LO, self.DT_HI, n))
        return np.clip(np.stack(cols, 1), 0.0, 1.0)

    def context_distribution(self, which='eval'):
        from scipy.stats import norm
        m, s = self._rho_params(which)
        z, w = _gauss_legendre_01(self.n_nodes[0])
        rho = self.RHO_LO + (self.RHO_HI - self.RHO_LO) * z
        w = w * norm.pdf((np.log(rho) - m) / s) / rho  # log-normal density
        w = w / w.sum()
        if self.contexts_dim == 1:
            return z.reshape(-1, 1), w
        z2, w2 = _truncnorm_rule(self.scen[which][1], self.DT_SD, self.DT_LO, self.DT_HI, self.n_nodes[1])
        C = np.stack([np.repeat(z, len(z2)), np.tile(z2, len(z))], 1)
        return C, np.outer(w, w2).ravel()


class Crop_Management_Shift(Crop_Management):
    """Strategy learned under the historical rainfall variability, regret under a severe-drought
    scenario (rain x 0.5). Irrigation buffers rainfall, so the shift regret is modest (~ 2 %)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        kw.setdefault('rain_factor_eval', 0.5)
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kw)


class Crop_Management_2C(Crop_Management):
    """Contexts (rainfall multiplier, temperature offset)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, contexts_dim=2, **kw)


class Crop_Management_2C_Shift(Crop_Management):
    """Two contexts; regret under a drier and warmer scenario (rain x 0.75, +1.5 degC)."""

    def __init__(self, noise_std=None, negate=False, bounds=None, **kw):
        kw.setdefault('rain_factor_eval', 0.75)
        kw.setdefault('dT_eval', 1.5)
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, contexts_dim=2, **kw)
