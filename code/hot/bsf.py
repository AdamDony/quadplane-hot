"""Backup safety filter (predictive lift-safety module): a robust backup manoeuvre is rolled out from the predicted
next state for a set of plants (vertices of the box plus samples); the network command is accepted if every rollout
satisfies the constraints and ends in the terminal set, otherwise the command is moved towards the backup command."""
import itertools, numpy as np
from . import plant as PL
DT = PL.DT; M, G, S, RHO = PL.M, PL.G, PL.S, PL.RHO
lo = {k: PL.BOX[k][0] for k in PL.KEYS}; hi = {k: PL.BOX[k][1] for k in PL.KEYS}
# backup law constants
M_B, K_GAM = 0.10, 2.0                   # target lift factor 1 + M_B - K_GAM*gamma
H_BACKUP = 200                           # backup horizon [steps] (10 s); the rollout must end in the settled neighbourhood
SUB = 2                                  # RK4 substeps per sample in the rollouts
GD_LIM = 0.30                            # sink-rate limit [rad/s] (17 deg/s); the policy is trained to 0.2
V_MIN = 4.0
KL = PL.KL_ROB                           # lift requirement: lift fraction >= 1 - KL
TERM = dict(lift=(0.93, 1.07), gam_abs=np.radians(3.0), gd_abs=0.05, al=(np.radians(-9.0), np.radians(0.5)), V=(5.0, 16.0), vd_abs=0.6)   # settled neighbourhood: the backup keeps the constraints from here (verified over 20 s)
FAST = ["kappa", "g", "lam_c", "lam_r", "beta_c", "beta_r"]
def plant_set(n_samples=256, seed=0, reduced=False):
    """vertices of the box (1024) or the reduced set (64 fast-parameter vertices x lift corners CL0, CLa = 256), plus samples"""
    if reduced:
        cols = {k: PL.BOX[k] if k in FAST + ["CL0"] else (PL.NOM[k],) for k in PL.KEYS}
        verts = np.array(list(itertools.product(*[cols[k] for k in PL.KEYS])))
    else: verts = np.array(list(itertools.product(*[PL.BOX[k] for k in PL.KEYS])))
    rng = np.random.default_rng(seed); smp = np.column_stack([rng.uniform(*PL.BOX[k], n_samples) for k in PL.KEYS])
    ctr = np.array([PL.NOM[k] for k in PL.KEYS]); half = np.array([0.5 * (PL.BOX[k][1] - PL.BOX[k][0]) for k in PL.KEYS])
    near = ctr + rng.uniform(-0.5, 0.5, (n_samples, len(PL.KEYS))) * half                 # interior samples within half the intervals about the identified plant
    allp = np.vstack([verts, near, ctr[None]]); return {k: allp[:, i].copy() for i, k in enumerate(PL.KEYS)}
def L_lb(V, al):
    qb = 0.5 * RHO * np.maximum(V, 1.0) ** 2; CLa_w = np.where(al >= 0, lo["CLa"], hi["CLa"])
    return qb * S * (lo["CL0"] + CLa_w * al)
K_GD, K_R = 0.4, 10.0                    # backup gains: level-off rate 0.4/s (slower than the slowest attitude loop, 0.67/s), rotor N per (rad/s) per step
LIFT_FLOOR = 1 - KL + 0.10               # guaranteed lift factor requested from the rotors in the backup (worst-case aero)
def gamma_dot(x, P):
    V, gam, th, Tc, Tr = x.T; al = th - gam
    L, D = PL.aero(np.maximum(V, 1.0), al, P); return (L + Tc * np.sin(al) + Tr * np.cos(al) - M * G * np.cos(gam)) / (M * np.maximum(V, 1.0))
AL_BK = (np.radians(-10.0), np.radians(1.0))  # angle-of-attack band of the backup pitch law
V_BAND = (6.0, 15.5)                          # airspeed corridor of the backup pusher loop
GD_REF = (-0.15, 0.15)                        # flight-path-rate reference band [rad/s]
TAU_LEAD = 0.75                               # lead of the backup pitch setpoint on the level-off [s]: the slowest attitude loop (1.5 s) would otherwise overshoot the angle of attack by (1.5 - tau) K_GD gamma
_VT = np.linspace(4.0, 18.0, 57); _TRIMS = [PL.trim(v, {k: PL.NOM[k] for k in PL.KEYS}) for v in _VT]; _AT = np.array([t[0][2] for t in _TRIMS]); _TCT = np.array([t[0][3] for t in _TRIMS])
K_V_B, A_MAX_B, K_C_B = 0.5, 1.0, 6.0        # backup pusher loop: acceleration reference k_V (V_corridor - V) saturated at +-a_max [m/s^2], N per (m/s^2) per step
def tc_trim(V): return np.interp(V, _VT, _TCT)
def alpha_ref(V): return np.clip(np.interp(V, _VT, _AT), AL_BK[0], AL_BK[1])   # nominal-trim angle of attack schedule (0 deg below the wing-borne speed, negative above)
def V_dot(x, P):
    V, gam, th, Tc, Tr = x.T; al = th - gam
    L, D = PL.aero(np.maximum(V, 1.0), al, P); return (Tc * np.cos(al) - D - Tr * np.sin(al) - M * G * np.sin(gam)) / M
def backup(x, u_prev, gd_meas, vd_meas):
    """backup command for a batch, from measured quantities only: pitch setpoint = flight path (with lead) + scheduled angle of
    attack, rotors regulate the measured flight-path rate to a level-off reference (integral action, rate-limited),
    pusher = the same kind of loop on the measured acceleration, whose reference is zero inside the airspeed corridor"""
    V, gam, th, Tc, Tr = x.T; al = th - gam
    r_ref = np.clip(-K_GD * gam, GD_REF[0], GD_REF[1])
    thsp = gam + TAU_LEAD * np.minimum(r_ref, 0.0) + alpha_ref(V)     # pitch leads a level-off from a climb so that slow attitude loops do not overshoot the angle of attack
    thsp = np.clip(thsp, u_prev[:, 0] - PL.DU[0], u_prev[:, 0] + PL.DU[0]); thsp = np.clip(thsp, PL.U_MIN[0], PL.U_MAX[0])
    Trc = u_prev[:, 2] + np.clip(K_R * (r_ref - gd_meas), -PL.DU[2], PL.DU[2]); Trc = np.clip(Trc, 0.0, PL.K_R)
    a_ref = np.clip(K_V_B * (np.clip(V, V_BAND[0], V_BAND[1]) - V), -A_MAX_B, A_MAX_B)      # zero inside the corridor: hold the current airspeed
    Tcc = u_prev[:, 1] + np.clip(K_C_B * (a_ref - vd_meas), -PL.DU[1], PL.DU[1]); Tcc = np.clip(Tcc, 0.0, PL.K_C)
    return np.column_stack([thsp, Tcc, Trc])
def constraints(x, P):
    """margins (>= 0 means satisfied) for a batch: lift, alpha, theta box, gamma box, sink rate, V_min; returns (n, 8)"""
    V, gam, th, Tc, Tr = x.T; al = th - gam
    L, D = PL.aero(np.maximum(V, 1.0), al, P); lift = L + Tc * np.sin(al) + Tr * np.cos(al)
    gd = (lift - M * G * np.cos(gam)) / (M * np.maximum(V, 1.0))
    return np.column_stack([lift / (M * G * np.cos(gam)) - (1 - KL), PL.AL_MAX - al, PL.TH_MAX - th, th - PL.TH_MIN, PL.GAM_MAX - gam, gam - PL.GAM_MIN, gd + GD_LIM, V - V_MIN])
def terminal(x, P):
    """margins of the settled neighbourhood: lift fraction band, small flight path and rate, alpha band, airspeed band, small acceleration"""
    V, gam, th, Tc, Tr = x.T; al = th - gam
    L, D = PL.aero(np.maximum(V, 1.0), al, P); lift = L + Tc * np.sin(al) + Tr * np.cos(al); lf = lift / (M * G * np.cos(gam)); gd = (lift - M * G * np.cos(gam)) / (M * np.maximum(V, 1.0))
    return np.column_stack([lf - TERM["lift"][0], TERM["lift"][1] - lf, TERM["gam_abs"] - np.abs(gam), TERM["gd_abs"] - np.abs(gd), al - TERM["al"][0], TERM["al"][1] - al, V - TERM["V"][0], TERM["V"][1] - V, TERM["vd_abs"] - np.abs(V_dot(x, P))])
EPS_CONS = np.array([0.4, np.radians(1.0), np.radians(1.0), 1.5, 4.0])    # one-step consistency tolerance on (V, gam, th, Tc, Tr)
MARGIN = np.array([0.02, np.radians(0.25), np.radians(0.5), np.radians(0.5), np.radians(1.0), np.radians(1.0), 0.02, 0.3])   # verification margins per constraint
class Filter:
    """backup safety filter over a finite plant set; plants inconsistent with the observed transitions are discarded
    (set-membership), which keeps the shift argument valid for every retained plant."""
    def __init__(self, n_samples=256, seed=0, horizon=H_BACKUP, reduced=False, consistency=True):
        self.P0 = plant_set(n_samples, seed, reduced); self.n0 = len(self.P0["CL0"]); self.H = horizon; self.consistency = consistency
        self.reset()
    def reset(self):
        self.alive = np.ones(self.n0, bool); self.stats = dict(checks=0, interventions=0, unverified_backup=0, steps=0); self.pending = None
    @property
    def P(self): return {k: v[self.alive] for k, v in self.P0.items()}
    @property
    def n(self): return int(self.alive.sum())
    def observe(self, x_prev, u_prev, x_now):
        """discard plants whose one-step prediction from (x_prev, u_prev) is inconsistent with the observed x_now"""
        if not self.consistency: return
        X = np.repeat(x_prev[None], self.n0, 0); U = np.repeat(u_prev[None], self.n0, 0)
        Xp = PL.step(X, U, self.P0); dev = np.abs(Xp - x_now[None]) / EPS_CONS[None]; ok = dev.max(1) <= 1.0
        if (ok & self.alive).sum() >= max(32, self.n0 // 10): self.alive &= ok
    def check(self, x, u_prev, u):
        P = self.P; n = self.n
        X = np.repeat(x[None], n, 0); U = np.repeat(u[None], n, 0)
        X = PL.step(X, U, P, sub=SUB); margin = (constraints(X, P) - MARGIN).min(); Uprev = U
        for k in range(self.H):
            if margin < 0: return False, margin
            tm = terminal(X, P).min()
            if tm >= 0: return True, min(margin, tm)          # every plant settled: the rest is covered by the settling property
            Ub = backup(X, Uprev, gamma_dot(X, P), V_dot(X, P)); X = PL.step(X, Ub, P, sub=SUB); Uprev = Ub; margin = min(margin, (constraints(X, P) - MARGIN).min())
        tm = terminal(X, P).min(); return (margin >= 0) and (tm >= 0), min(margin, tm)
    def __call__(self, x, u_prev, u_nn, gd_meas, vd_meas, n_bisect=3):
        self.stats["checks"] += 1; self.stats["steps"] += 1
        ok, _ = self.check(x, u_prev, u_nn)
        if ok: return u_nn.copy(), False
        self.stats["interventions"] += 1
        # first choice: the network's increment scaled down towards holding the previous command (mildest intervention)
        if self.check(x, u_prev, u_prev)[0]:
            lam_lo, lam_hi = 0.0, 1.0
            for _ in range(n_bisect):
                lam = 0.5 * (lam_lo + lam_hi); u = u_prev + lam * (u_nn - u_prev)
                if self.check(x, u_prev, u)[0]: lam_lo = lam
                else: lam_hi = lam
            return u_prev + lam_lo * (u_nn - u_prev), True
        # second choice: the segment towards the backup command; last resort the backup itself (Lemma: recursive feasibility)
        ub = backup(x[None], u_prev[None], np.array([gd_meas]), np.array([vd_meas]))[0]
        okb, mb = self.check(x, u_prev, ub)
        if not okb: self.stats["unverified_backup"] += 1; return ub, True
        lam_lo, lam_hi = 0.0, 1.0
        for _ in range(n_bisect):
            lam = 0.5 * (lam_lo + lam_hi); u = ub + lam * (u_nn - ub)
            if self.check(x, u_prev, u)[0]: lam_lo = lam
            else: lam_hi = lam
        return ub + lam_lo * (u_nn - ub), True
