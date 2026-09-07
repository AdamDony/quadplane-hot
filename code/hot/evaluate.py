"""Closed-loop evaluation of a controller on a plant: metrics, cells (nominal, cross-plant, out-of-box, gust, noise)."""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hot import plant as PL, bsf
NOISE = np.array([0.3, np.radians(0.5), np.radians(0.3), 0.6, 2.4])
P_HOVER, P_CRUISE = 1192.8, 380.0; T_C_CRUISE = PL.trim(15.0, {k: PL.NOM[k] for k in PL.KEYS})[1][1]
C_R = P_HOVER / (PL.M * PL.G) ** 1.5; C_C = P_CRUISE / T_C_CRUISE      # rotor power: momentum-theory 3/2 law through the hover point; pusher: linear through the cruise point
def power(Tc, Tr): return C_R * np.maximum(Tr, 0) ** 1.5 + C_C * np.maximum(Tc, 0)
def gust_series(n, rng, sigma):
    if sigma <= 0: return np.zeros(n)
    c = rng.uniform(-sigma, sigma); a = np.exp(-PL.DT); w = rng.normal(0, sigma * np.sqrt(1 - a * a), n); g = np.zeros(n); z = 0.0
    for j in range(n): z = a * z + w[j]; g[j] = c + z
    return g
def run_episode(ctrl, P1, rho0, rho_t, n_steps=400, gust_sigma=0.0, noise_scale=0.0, seed=0, x0=None, u0=None):
    """P1: dict of length-1 arrays (one plant). Returns log dict with arrays and metrics."""
    rng = np.random.default_rng(seed); nom = {k: PL.NOM[k] for k in PL.KEYS}
    if x0 is None: x0, u0 = PL.trim(rho0, nom)
    x = x0.copy(); ctrl.reset(x0, u0, rho_t); gust = gust_series(n_steps, rng, gust_sigma)
    X, U, LF, GD, T = [], [], [], [], []; t_target = None; viol = np.zeros(8); t0 = time.time()
    for k in range(n_steps):
        gd_true = bsf.gamma_dot(x[None], P1)[0]
        x_meas = x + rng.normal(0, 1, 5) * NOISE * noise_scale; gd_meas = gd_true + rng.normal(0, 0.02 * noise_scale)
        vd_meas = bsf.V_dot(x[None], P1)[0] + rng.normal(0, 0.1 * noise_scale)
        u = ctrl.step(x_meas, gd_meas, rho_t, vd_meas)
        x = PL.step(x[None], u[None], P1, gust=np.array([gust[k]]))[0]
        c = bsf.constraints(x[None], P1)[0]; viol += (c < 0); X.append(x.copy()); U.append(u.copy()); LF.append(PL.lift_fraction(x[None], P1)[0]); GD.append(gd_true)
        if t_target is None and abs(x[0] - rho_t) <= 0.3: t_target = (k + 1) * PL.DT
    X = np.array(X); U = np.array(U); LF = np.array(LF)
    energy = float(np.sum(power(X[:, 3], X[:, 4]) * PL.DT) / 3600.0)             # Wh with the identified power model
    m = dict(t_target=t_target if t_target is not None else float("nan"), reached=t_target is not None, energy_Wh=energy, lf_min=float(LF.min()), al_max_deg=float(np.degrees((X[:, 2] - X[:, 1]).max())),
             gam_min_deg=float(np.degrees(X[:, 1].min())), gam_max_deg=float(np.degrees(X[:, 1].max())), sink_min=float(min(GD)), viol_ticks=int((viol > 0).sum()), viol_pct=float(100 * np.mean([bsf.constraints(xx[None], P1)[0].min() < 0 for xx in X])),
             wall_ms_per_step=1e3 * (time.time() - t0) / n_steps)
    if hasattr(ctrl, "F") and ctrl.F is not None: m.update(interventions=ctrl.F.stats["interventions"], unverified=ctrl.F.stats["unverified_backup"], plants_alive=ctrl.F.n)
    return dict(X=X, U=U, LF=LF, GD=np.array(GD), gust=gust, metrics=m)
def one_plant(P, i): return {k: v[i:i + 1] for k, v in P.items()}
