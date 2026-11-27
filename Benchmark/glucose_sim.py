"""Vectorised numpy port of the UVA/Padova type-1-diabetes patient model of simglucose
(https://github.com/jxx123/simglucose, MIT licence, (c) Jinyu Xie; model of Dalla Man et al. 2007/2014,
parameters of the 30 virtual patients in params/vpatient_params.csv).

The right-hand side is a line-by-line copy of simglucose.patient.t1dpatient.T1DPatient.model,
including the meal-dependent gastric-emptying rate (kgut), the clamps of the states at zero and
the eating rate of 5 g/min.  Instead of scipy's adaptive dopri5 per one-minute action interval
it integrates a whole batch of independent scenarios at once with classical RK4 (step 0.5 min,
inputs piecewise constant over each step), which agrees with simglucose to < 0.05 mg/dL
(checked in Benchmark/real_world_tools/check_glucose_port.py).

One scenario = the patient starts in its basal steady state, eats a meal of `cho` grams starting
at t = 0 (5 g/min), and receives a bolus of `dose` units delivered at a constant rate during the
minute [t_bolus, t_bolus + 1) on top of the basal insulin u2ss * BW / 6000 U/min.
"""
import os

import numpy as np

EAT_RATE = 5.0  # g/min, as in simglucose
PARAM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'simglucose_vpatient_params.csv')


def load_patient(name='adult#001', param_file=PARAM_FILE):
    import pandas as pd
    df = pd.read_csv(param_file)
    row = df.loc[df.Name == name].squeeze()
    if row.empty:
        raise ValueError(f'unknown patient {name}')
    p = {k: float(row[k]) for k in row.index if k not in ('Name', 'patient_history')}
    p['x0'] = row.iloc[2:15].to_numpy(dtype=float)  # steady-state initial condition
    p['name'] = name
    return p


def _rhs(x, d_mgmin, ins_Umin, Dbar, P):
    """Batch right-hand side. x: (n, 13); d_mgmin, ins_Umin, Dbar: (n,)."""
    dx = np.empty_like(x)
    insulin = ins_Umin * 6000.0 / P['BW']
    qsto = x[:, 0] + x[:, 1]
    dx[:, 0] = -P['kmax'] * x[:, 0] + d_mgmin
    with np.errstate(divide='ignore', invalid='ignore'):
        aa = 5.0 / (2.0 * Dbar * (1.0 - P['b']))
        cc = 5.0 / (2.0 * Dbar * P['d'])
        kgut = P['kmin'] + (P['kmax'] - P['kmin']) / 2.0 * (
            np.tanh(aa * (qsto - P['b'] * Dbar)) - np.tanh(cc * (qsto - P['d'] * Dbar)) + 2.0)
    kgut = np.where(Dbar > 0, kgut, P['kmax'])
    dx[:, 1] = P['kmax'] * x[:, 0] - x[:, 1] * kgut
    dx[:, 2] = kgut * x[:, 1] - P['kabs'] * x[:, 2]
    Rat = P['f'] * P['kabs'] * x[:, 2] / P['BW']
    EGPt = P['kp1'] - P['kp2'] * x[:, 3] - P['kp3'] * x[:, 8]
    Et = np.where(x[:, 3] > P['ke2'], P['ke1'] * (x[:, 3] - P['ke2']), 0.0)
    dx[:, 3] = np.maximum(EGPt, 0.0) + Rat - P['Fsnc'] - Et - P['k1'] * x[:, 3] + P['k2'] * x[:, 4]
    dx[:, 3] *= (x[:, 3] >= 0)
    Vmt = P['Vm0'] + P['Vmx'] * x[:, 6]
    Uidt = Vmt * x[:, 4] / (P['Km0'] + x[:, 4])
    dx[:, 4] = (-Uidt + P['k1'] * x[:, 3] - P['k2'] * x[:, 4]) * (x[:, 4] >= 0)
    dx[:, 5] = (-(P['m2'] + P['m4']) * x[:, 5] + P['m1'] * x[:, 9] + P['ka1'] * x[:, 10]
                + P['ka2'] * x[:, 11]) * (x[:, 5] >= 0)
    It = x[:, 5] / P['Vi']
    dx[:, 6] = -P['p2u'] * x[:, 6] + P['p2u'] * (It - P['Ib'])
    dx[:, 7] = -P['ki'] * (x[:, 7] - It)
    dx[:, 8] = -P['ki'] * (x[:, 8] - x[:, 7])
    dx[:, 9] = (-(P['m1'] + P['m30']) * x[:, 9] + P['m2'] * x[:, 5]) * (x[:, 9] >= 0)
    dx[:, 10] = (insulin - (P['ka1'] + P['kd']) * x[:, 10]) * (x[:, 10] >= 0)
    dx[:, 11] = (P['kd'] * x[:, 10] - P['ka2'] * x[:, 11]) * (x[:, 11] >= 0)
    dx[:, 12] = (-P['ksc'] * x[:, 12] + P['ksc'] * x[:, 3]) * (x[:, 12] >= 0)
    return dx


def simulate_schedule(P, meal_times, cho, bolus_times, dose, t_start=-60.0, t_end=360.0, h=0.5, record_every=1.0,
                      mean_risk_from=None):
    """Batch simulation with several meals and boluses (same times for the whole batch).

    meal_times (k_m,) min; cho (n, k_m) g; bolus_times (n, k_b) or (k_b,) min; dose (n, k_b) U.
    Meals are eaten at 5 g/min from their start time (they must not overlap). As in simglucose, the
    gastric-emptying reference Dbar of a meal is the stomach content at its start plus the food taken.
    Returns (t, BG) with BG (n, T) in mg/dL, or, if mean_risk_from is given, the mean Kovatchev risk
    of the samples with t >= mean_risk_from (n,) without storing the trajectories.
    """
    cho = np.atleast_2d(np.asarray(cho, float))
    dose = np.atleast_2d(np.asarray(dose, float))
    n = max(cho.shape[0], dose.shape[0])
    cho = np.broadcast_to(cho, (n, cho.shape[1]))
    dose = np.broadcast_to(dose, (n, dose.shape[1]))
    bolus_times = np.broadcast_to(np.atleast_2d(np.asarray(bolus_times, float)), dose.shape)
    meal_times = np.asarray(meal_times, float)
    basal = P['u2ss'] * P['BW'] / 6000.0  # U/min
    x = np.tile(P['x0'], (n, 1))
    last_qsto = np.zeros(n)
    steps = int(round((t_end - t_start) / h))
    rec = int(round(record_every / h))
    out_t, out_bg = [], []
    active = -1
    for k in range(steps):
        t = t_start + k * h
        minute = np.floor(t)
        # the meal whose eating period may contain this minute (latest meal started at or before it)
        started = np.nonzero(meal_times <= minute)[0]
        m = started[-1] if len(started) else -1
        if m != active and m >= 0:
            last_qsto = x[:, 0] + x[:, 1]  # stomach content when the new meal starts
            active = m
        if m >= 0:
            elapsed = minute - meal_times[m]
            eaten_before = np.clip(EAT_RATE * elapsed, 0.0, cho[:, m])
            this_min = np.clip(cho[:, m] - eaten_before, 0.0, EAT_RATE)
            Dbar = last_qsto + (eaten_before + this_min) * 1000.0
        else:
            this_min = np.zeros(n)
            Dbar = np.zeros(n)
        d_mgmin = this_min * 1000.0
        overlap = np.clip(np.minimum(t + h, bolus_times + 1.0) - np.maximum(t, bolus_times), 0.0, h) / h
        ins = basal + (dose * overlap).sum(1)
        k1 = _rhs(x, d_mgmin, ins, Dbar, P)
        k2 = _rhs(x + 0.5 * h * k1, d_mgmin, ins, Dbar, P)
        k3 = _rhs(x + 0.5 * h * k2, d_mgmin, ins, Dbar, P)
        k4 = _rhs(x + h * k3, d_mgmin, ins, Dbar, P)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (k + 1) % rec == 0:
            if mean_risk_from is None:
                out_t.append(t + h)
                out_bg.append(x[:, 3] / P['Vg'])
            elif t + h >= mean_risk_from:
                out_t.append(t + h)
                acc = risk_index(x[:, 3] / P['Vg'])
                out_bg = acc if isinstance(out_bg, list) else out_bg + acc
    if mean_risk_from is not None:
        return out_bg / len(out_t)
    return np.asarray(out_t), np.stack(out_bg, 1)


def simulate(P, cho, dose, t_bolus, t_start=-60.0, t_end=360.0, h=0.5, record_every=1.0):
    """One meal of `cho` g at t = 0 and one bolus of `dose` U at `t_bolus` min (arrays of length n)."""
    cho = np.atleast_1d(np.asarray(cho, float))
    dose = np.atleast_1d(np.asarray(dose, float))
    t_bolus = np.atleast_1d(np.asarray(t_bolus, float))
    n = max(len(cho), len(dose), len(t_bolus))
    cho, dose, t_bolus = (np.broadcast_to(a, (n,)) for a in (cho, dose, t_bolus))
    return simulate_schedule(P, [0.0], cho[:, None], t_bolus[:, None], dose[:, None], t_start, t_end, h,
                             record_every)


def risk_index(bg):
    """Kovatchev blood-glucose risk (LBGI + HBGI contribution of each sample), as in simglucose."""
    bg = np.maximum(bg, 1.0)
    fbg = 1.509 * (np.log(bg) ** 1.084 - 5.381)
    return 10.0 * fbg ** 2
