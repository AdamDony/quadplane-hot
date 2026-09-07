"""Identified five-state longitudinal model, parameter box and a batched numpy simulator."""
import json, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
IDENT = json.load(open(os.path.join(H, "..", "ident.json")))
G, RHO, M, S = 9.81, 1.225, 8.0, 2.0 * 0.292
DT = 0.05
_a, _r = IDENT["aero"], IDENT["rotor"]
NOM = dict(CL0=_a["CL0"], CLa=_a["CLa"], CD0=_a["CD0"], k=_a["k"], kappa=2.0, g=1.0, lam_c=2.0, lam_r=1 / 0.15, beta_c=1.0, beta_r=1.0)
K_C, K_R = _a["k_c"], _r["k_r"]                   # thrust command ranges [N]
BOX = dict(CL0=(NOM["CL0"] - 2 * _a["CL0_std"], NOM["CL0"] + 2 * _a["CL0_std"]), CLa=(0.75 * NOM["CLa"], 1.25 * NOM["CLa"]),
           CD0=(NOM["CD0"] - 0.044, NOM["CD0"] + 0.044), k=(0.75 * NOM["k"], 1.25 * NOM["k"]),
           kappa=(1 / 1.5, 1 / 0.2), g=(0.9, 1.0), lam_c=(1 / 0.75, 1 / 0.25), lam_r=(1 / 0.225, 1 / 0.075),
           beta_c=(1 - 2 * 3.3 / K_C, 1 + 2 * 3.3 / K_C), beta_r=(0.85, 1.15))
KEYS = list(BOX.keys())
U_MIN = np.array([-np.radians(30), 0.0, 0.0]); U_MAX = np.array([np.radians(30), K_C, K_R]); DU = np.array([np.radians(4.0), 0.75, 4.0])   # per-step bounds: 80 deg/s setpoint, 15 N/s pusher, 80 N/s rotors
TH_MIN, TH_MAX, AL_MAX, GAM_MIN, GAM_MAX = np.radians(-15), np.radians(20), np.radians(5), np.radians(-15), np.radians(15)
KL_NOM, KL_ROB = 0.15, 0.25
RHO0, RHOC = 6.0, 15.0                            # hover-exit and cruise airspeeds of the hand-off
def sample_box(n, rng, scale=1.0):
    """n parameter vectors uniform in the box (scale > 1 enlarges the box about its centre for out-of-box tests)"""
    P = {}
    for kk in KEYS:
        lo, hi = BOX[kk]; c, h = 0.5 * (lo + hi), 0.5 * (hi - lo) * scale; P[kk] = rng.uniform(c - h, c + h, n)
    return P
def nominal(n):
    return {kk: np.full(n, NOM[kk]) for kk in KEYS}
def vertices():
    import itertools
    out = []
    for c in itertools.product(*[BOX[kk] for kk in KEYS]): out.append(dict(zip(KEYS, c)))
    return out
AL_STALL, AL_NEG = np.radians(10.0), np.radians(-14.0)      # post-stall model of the evaluation plant: lift curve folds beyond the stall angles
K_POST = 2.0                                                  # post-stall lift-curve slope [1/rad] (negative), drag rises with the excess angle
def aero(V, al, P):
    """lift and drag; linear polar within [AL_NEG, AL_STALL], folded lift curve and rising drag beyond (evaluation plant)"""
    qb = 0.5 * RHO * V ** 2; al_c = np.clip(al, AL_NEG, AL_STALL); ex = np.abs(al - al_c)
    CL = np.where(al > AL_STALL, P["CL0"] + P["CLa"] * AL_STALL - K_POST * ex, np.where(al < AL_NEG, P["CL0"] + P["CLa"] * AL_NEG + K_POST * ex, P["CL0"] + P["CLa"] * al))
    CD = P["CD0"] + P["k"] * CL ** 2 + 1.5 * ex ** 2
    return qb * S * CL, qb * S * CD
def f(x, u, P, w=0.0):
    """batched vector field; x (n,5), u (n,3), P dict of (n,) arrays, w vertical gust [m/s]: it changes the aerodynamic
    angle of attack of the wing (lift and drag) by arctan(w/V); the attitude loop and the thrust directions do not see it"""
    V, gam, th, Tc, Tr = x.T; thsp, Tcc, Trc = u.T
    Vs = np.maximum(V, 1.0); al = th - gam
    L, D = aero(Vs, al + np.arctan2(w, Vs), P)
    dV = (Tc * np.cos(al) - D - Tr * np.sin(al) - M * G * np.sin(gam)) / M
    dg = (L + Tc * np.sin(al) + Tr * np.cos(al) - M * G * np.cos(gam)) / (M * Vs)
    dth = P["kappa"] * (P["g"] * thsp - th)
    dTc = P["lam_c"] * (P["beta_c"] * Tcc - Tc); dTr = P["lam_r"] * (P["beta_r"] * Trc - Tr)
    return np.stack([dV, dg, dth, dTc, dTr], 1)
def step(x, u, P, dt=DT, sub=5, gust=None):
    """RK4 step with a vertical gust w [m/s] entering as an angle-of-attack perturbation atan(w/V)"""
    h = dt / sub; xx = x.copy(); w = 0.0 if gust is None else gust
    for _ in range(sub):
        fg = lambda y: f(y, u, P, w)
        k1 = fg(xx); k2 = fg(xx + h / 2 * k1); k3 = fg(xx + h / 2 * k2); k4 = fg(xx + h * k3)
        xx = xx + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return xx
def lift_fraction(x, P):
    V, gam, th, Tc, Tr = x.T; al = th - gam; L, _ = aero(np.maximum(V, 1.0), al, P)
    return (L + Tc * np.sin(al) + Tr * np.cos(al)) / (M * G * np.cos(gam))
def trim(V, P, T_idle=10.0, al_cap=0.0, gam=0.0):
    """lift-sharing trim of a single plant dict of scalars: alpha at the cap unless the wing alone exceeds the weight"""
    from scipy.optimize import brentq
    def resid(al):
        L, D = aero(V, al, P); A = np.array([[np.cos(al), -np.sin(al)], [np.sin(al), np.cos(al)]])
        Tc, Tr = np.linalg.solve(A, [D + M * G * np.sin(gam), M * G * np.cos(gam) - L]); return Tr - T_idle
    al = al_cap
    if resid(al_cap) < 0: al = brentq(resid, np.radians(-15), al_cap)
    L, D = aero(V, al, P); A = np.array([[np.cos(al), -np.sin(al)], [np.sin(al), np.cos(al)]])
    Tc, Tr = np.linalg.solve(A, [D + M * G * np.sin(gam), M * G * np.cos(gam) - L]); Tr = max(Tr, T_idle)
    th = al + gam; return np.array([V, gam, th, Tc, Tr]), np.array([th / P["g"], Tc / P["beta_c"], Tr / P["beta_r"]])
