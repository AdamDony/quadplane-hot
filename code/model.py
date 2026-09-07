"""5-state longitudinal quadplane model with PX4 attitude-loop lag and actuator lags.
x = [V, gamma, theta, Tc, Tr],  u = [theta_sp, Tc_cmd, Tr_cmd]  (thrusts in N, angles in rad)."""
import json, numpy as np
from scipy.optimize import brentq
G = 9.81; RHO = 1.225; M = 8.0; S = 2.0 * 0.292
import os; IDENT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ident.json")))
a = IDENT["aero"]; r = IDENT["rotor"]
NOM = dict(CL0=a["CL0"], CLa=a["CLa"], CD0=a["CD0"], k=a["k"], k_c=a["k_c"], k_r=r["k_r"], tau_c=0.5, tau_r=0.15,
           kappa=1 / 0.5, g=0.65, c_th=0.0, beta_c=1.0, beta_r=1.0)
# half-widths of the parameter box (identified: 2 sigma; assumed: stated)
HALF = dict(CL0=2 * a["CL0_std"], CLa=0.25 * a["CLa"], CD0=2 * 0.022, k=0.25 * a["k"])   # aero box (norm-bounded); CD0: 2 x HAC std
# polytopic (interval) parameters: attitude-loop rate/gain, actuator rates 1/tau, thrust-map gain errors beta = k_true / k_hat
LAMC_INT = (1 / 0.75, 1 / 0.25); LAMR_INT = (1 / 0.225, 1 / 0.075)
BETAC_INT = (1 - 2 * 3.3 / a["k_c"], 1 + 2 * 3.3 / a["k_c"]); BETAR_INT = (0.85, 1.15)
KAPPA_INT = (1 / 1.5, 1 / 0.2)     # attitude-loop rate 1/tau, tau in [0.2, 1.5] s
GAIN_INT = (0.35, 1.0)             # attitude-loop DC gain
U_MIN = np.array([-np.radians(30), 0.0, 0.0]); U_MAX = np.array([np.radians(30), NOM["k_c"], NOM["k_r"]])
T_IDLE = 10.0                      # rotor idle thrust kept during the certified transition (N)
ALPHA_CAP = np.radians(0.0)        # trim angle-of-attack cap (body axes), 5 deg below the barrier
# safety bounds
QBAR = np.radians(40.0)            # commanded pitch-rate bound
TH_MIN, TH_MAX = np.radians(-15.0), np.radians(20.0)
AL_MAX = np.radians(5.0)           # angle-of-attack bound (body axes): inside the range verified unstalled in cruise
GAM_MIN, GAM_MAX = np.radians(-15.0), np.radians(15.0)
KL = 0.15                          # allowed transient lift deficit (15 % of weight)

def aero(V, al, p):
    qb = 0.5 * RHO * V ** 2; CL = p["CL0"] + p["CLa"] * al
    return qb * S * CL, qb * S * (p["CD0"] + p["k"] * CL ** 2)

def f(x, u, p):
    V, gam, th, Tc, Tr = x; thsp, Tcc, Trc = u
    al = th - gam; Vs = max(V, 1.0); L, D = aero(Vs, al, p)
    return np.array([(Tc * np.cos(al) - D - Tr * np.sin(al) - M * G * np.sin(gam)) / M,
                     (L + Tc * np.sin(al) + Tr * np.cos(al) - M * G * np.cos(gam)) / (M * Vs),
                     p["kappa"] * (p["g"] * thsp - th) + p["c_th"],
                     (p["beta_c"] * Tcc - Tc) / p["tau_c"], (p["beta_r"] * Trc - Tr) / p["tau_r"]])

def trim(V, p, gam_ref=0.0):
    """Lift-sharing trim: alpha = min(alpha_cap, alpha_wb), rotors carry the deficit (>= T_IDLE)."""
    def resid_wb(al):
        L, D = aero(V, al, p); A = np.array([[np.cos(al), -np.sin(al)], [np.sin(al), np.cos(al)]])
        Tc, Tr = np.linalg.solve(A, [D + M * G * np.sin(gam_ref), M * G * np.cos(gam_ref) - L]); return Tr - T_IDLE
    al = ALPHA_CAP
    if resid_wb(ALPHA_CAP) < 0:      # wing (plus idle rotors) would exceed weight at the cap -> reduce alpha
        al = brentq(resid_wb, np.radians(-15), ALPHA_CAP)
    L, D = aero(V, al, p); A = np.array([[np.cos(al), -np.sin(al)], [np.sin(al), np.cos(al)]])
    Tc, Tr = np.linalg.solve(A, [D + M * G * np.sin(gam_ref), M * G * np.cos(gam_ref) - L]); Tr = max(Tr, T_IDLE)
    th = al + gam_ref; thsp = (th - p["c_th"] / p["kappa"]) / p["g"]
    return np.array([V, gam_ref, th, Tc, Tr]), np.array([thsp, Tc, Tr])

def jac(x, u, p, h=1e-5):
    n, m = len(x), len(u); A = np.zeros((n, n)); B = np.zeros((n, m))
    for i in range(n):
        d = np.zeros(n); d[i] = h; A[:, i] = (f(x + d, u, p) - f(x - d, u, p)) / (2 * h)
    for i in range(m):
        d = np.zeros(m); d[i] = h; B[:, i] = (f(x, u + d, p) - f(x, u - d, p)) / (2 * h)
    return A, B

# barrier functions h_j(x) >= 0 (order-1 or order-2 in the inputs)
def barriers(x, p):
    V, gam, th, Tc, Tr = x; al = th - gam; L, D = aero(max(V, 1.0), al, p)
    return dict(th_hi=TH_MAX - th, th_lo=th - TH_MIN, al=AL_MAX - al, gam_hi=GAM_MAX - gam, gam_lo=gam - GAM_MIN,
                lift=L + Tr * np.cos(al) - (1 - KL) * M * G * np.cos(gam))
BARRIER_ORDER = dict(th_hi=1, th_lo=1, al=1, gam_hi=2, gam_lo=2, lift=1)

def grad_lift(x, p):
    V, gam, th, Tc, Tr = x; al = th - gam; Vs = max(V, 1.0); qb = 0.5 * RHO * Vs ** 2; CL = p["CL0"] + p["CLa"] * al
    dV = RHO * Vs * S * CL; dal = qb * S * p["CLa"] - Tr * np.sin(al)
    return np.array([dV, -dal + (1 - KL) * M * G * np.sin(gam), dal, 0.0, np.cos(al)])
def grad_h(x, p, name):
    e = np.eye(5)
    return {"th_hi": -e[2], "th_lo": e[2], "al": e[1] - e[2], "gam_hi": -e[1], "gam_lo": e[1]}[name] if name != "lift" else grad_lift(x, p)

def wing_borne_speed(p, al=ALPHA_CAP):
    return np.sqrt(2 * M * G / (RHO * S * (p["CL0"] + p["CLa"] * al)))

if __name__ == "__main__":
    print("nominal:", {k: round(v, 4) for k, v in NOM.items()})
    print("half-widths:", {k: round(v, 4) for k, v in HALF.items()})
    print(f"wing-borne speed at alpha_cap: {wing_borne_speed(NOM):.2f} m/s")
    for V in [6, 8, 10, 12, 13, 14, 15]:
        xt, ut = trim(V, NOM)
        print(f"V={V:4.1f}: alpha={np.degrees(xt[2]):5.2f} deg, Tc={xt[3]:6.2f} N, Tr={xt[4]:6.2f} N, theta_sp={np.degrees(ut[0]):5.2f} deg, resid={np.abs(f(xt, ut, NOM)).max():.1e}")
        A, B = jac(xt, ut, NOM); ev = np.linalg.eigvals(A)
        print("      open-loop eig:", np.round(np.sort_complex(ev), 3))
