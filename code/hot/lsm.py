"""Lift-safety module: one-step interval enclosure over the parameter box, linear sufficient rows, QP projection."""
import numpy as np
from scipy.optimize import linprog, minimize
from . import plant as PL
DT = PL.DT; M, G, S, RHO = PL.M, PL.G, PL.S, PL.RHO
TAU = 0.5                       # look-ahead of the flight-path barriers [s]
GD_LIM = 0.15                   # sink/climb-rate barrier: |gamma_dot| <= GD_LIM rad/s (8.6 deg/s)
lo = {k: PL.BOX[k][0] for k in PL.KEYS}; hi = {k: PL.BOX[k][1] for k in PL.KEYS}
TH_C = [(np.exp(-k * DT), (1 - np.exp(-k * DT)) * g) for k in PL.BOX["kappa"] for g in PL.BOX["g"]]
TC_C = [(np.exp(-l * DT), (1 - np.exp(-l * DT)) * b) for l in PL.BOX["lam_c"] for b in PL.BOX["beta_c"]]
TR_C = [(np.exp(-l * DT), (1 - np.exp(-l * DT)) * b) for l in PL.BOX["lam_r"] for b in PL.BOX["beta_r"]]
def I(a, b): return (min(a, b), max(a, b))
def imul(a, b): c = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]; return (min(c), max(c))
def iadd(*xs): return (sum(x[0] for x in xs), sum(x[1] for x in xs))
def ineg(a): return (-a[1], -a[0])
def icos(a): m = min(abs(a[0]), abs(a[1])) if a[0] * a[1] > 0 else 0.0; return (np.cos(max(abs(a[0]), abs(a[1]))), np.cos(m))
def isin(a): return (np.sin(a[0]), np.sin(a[1]))
def iaero(Vi, ali):
    qb = (0.5 * RHO * Vi[0] ** 2, 0.5 * RHO * Vi[1] ** 2)
    CL = iadd((lo["CL0"], hi["CL0"]), imul((lo["CLa"], hi["CLa"]), ali)); L = imul(qb, (S * CL[0], S * CL[1]))
    CL2 = (0.0, max(CL[0] ** 2, CL[1] ** 2)) if CL[0] < 0 < CL[1] else (min(CL[0] ** 2, CL[1] ** 2), max(CL[0] ** 2, CL[1] ** 2))
    CD = iadd((lo["CD0"], hi["CD0"]), imul((lo["k"], hi["k"]), CL2)); D = imul(qb, (S * CD[0], S * CD[1])); return L, D
def irates(Vi, gi, ali, Tci, Tri):
    L, D = iaero(Vi, ali); ca, sa = icos(ali), isin(ali)
    Vd = iadd(imul(Tci, ca), ineg(D), ineg(imul(Tri, sa)), ineg((M * G * np.sin(gi[0]), M * G * np.sin(gi[1]))))
    num = iadd(L, imul(Tci, sa), imul(Tri, ca), ineg((M * G * icos(gi)[1], M * G * icos(gi)[0])))
    gd = imul(num, (1 / (M * Vi[1]), 1 / (M * Vi[0]))); return (Vd[0] / M, Vd[1] / M), gd
def hrob(x, KL):
    V, gam, th, Tc, Tr = x; al = th - gam; qb = 0.5 * RHO * max(V, 1.0) ** 2
    CLa_w = lo["CLa"] if al >= 0 else hi["CLa"]
    return qb * S * (lo["CL0"] + CLa_w * al) + Tr * np.cos(al) - (1 - KL) * M * G * np.cos(gam)
def admissible(u_prev):
    return [(max(PL.U_MIN[i], u_prev[i] - PL.DU[i]), min(PL.U_MAX[i], u_prev[i] + PL.DU[i])) for i in range(3)]
def fast_intervals(x, u_prev, TCc=None, TRc=None):
    TCc = TC_C if TCc is None else TCc; TRc = TR_C if TRc is None else TRc
    V, gam, th, Tc, Tr = x; ths, tcs, trs = admissible(u_prev)
    thp = [a * th + c * sp for a, c in TH_C for sp in ths]; tcp = [a * Tc + c * cmd for a, c in TCc for cmd in tcs]; trp = [a * Tr + c * cmd for a, c in TRc for cmd in trs]
    return I(min(thp + [th]), max(thp + [th])), I(min(tcp + [Tc]), max(tcp + [Tc])), I(min(trp + [Tr]), max(trp + [Tr]))
def enclosure(x, u_prev, iters=6, TCc=None, TRc=None):
    V, gam, th, Tc, Tr = x; thi, tci, tri = fast_intervals(x, u_prev, TCc, TRc); Vi, gi = (V, V), (gam, gam); Vd = gd = (0.0, 0.0)
    for _ in range(iters):
        ali = iadd(thi, ineg(gi)); Vd, gd = irates((max(Vi[0], 1.0), Vi[1]), gi, ali, tci, tri)
        Vn = (V + DT * min(0.0, Vd[0]) * 1.05, V + DT * max(0.0, Vd[1]) * 1.05); gn = (gam + DT * min(0.0, gd[0]) * 1.05, gam + DT * max(0.0, gd[1]) * 1.05)
        if Vn[0] >= Vi[0] - 1e-9 and Vn[1] <= Vi[1] + 1e-9 and gn[0] >= gi[0] - 1e-9 and gn[1] <= gi[1] + 1e-9: break
        Vi, gi = (min(Vi[0], Vn[0]), max(Vi[1], Vn[1])), (min(gi[0], gn[0]), max(gi[1], gn[1]))
    return Vd, gd, thi, tci, tri
def rows(x, u_prev, KL=PL.KL_ROB, eta=0.5, h_floor=3.0, gains=None):
    """A u <= b: alpha barrier, theta box, lift barrier (on h - h_floor), input box and step bounds; gains = (beta_c_int, beta_r_int)"""
    TCc, TRc = gain_corners(*gains) if gains is not None else (TC_C, TR_C)
    V, gam, th, Tc, Tr = x; al = th - gam
    Vd, gd, thi, tci, tri = enclosure(x, u_prev, TCc=TCc, TRc=TRc)
    Vp_min = V + DT * min(0.0, Vd[0]); gp = (gam + DT * min(0.0, gd[0]), gam + DT * max(0.0, gd[1]))
    cg = np.cos(min(abs(gp[0]), abs(gp[1])) if gp[0] * gp[1] > 0 else 0.0)
    A, b = [], []
    allowed = PL.AL_MAX - (1 - eta) * (PL.AL_MAX - al)
    for a, c in TH_C:
        A.append([c, 0, 0]); b.append(allowed + gp[0] - a * th)
        A.append([c, 0, 0]); b.append(PL.TH_MAX - a * th); A.append([-c, 0, 0]); b.append(a * th - PL.TH_MIN)
    h1 = hrob(x, KL) - h_floor; target = (1 - eta) * h1 + h_floor; qb = 0.5 * RHO * max(Vp_min, 1.0) ** 2
    ali = iadd(thi, ineg(gp)); cA = np.cos(max(abs(ali[0]), abs(ali[1])))
    for CLa_w in (lo["CLa"], hi["CLa"]):
        for a, c in TH_C:
            for bb, d in TRc:
                A.append([-qb * S * CLa_w * c, 0, -d * cA]); b.append(qb * S * (lo["CL0"] + CLa_w * (a * th - gp[1])) + bb * Tr * cA - (1 - KL) * M * G * cg - target)
    # flight-path barriers of relative degree two: h3 = gam - GAM_MIN + tau*gd, h4 = GAM_MAX - gam - tau*gd,
    # enforced on the one-step prediction through the lift numerator, linear in (th_sp, Tr_c)
    Vp = (V + DT * min(0.0, Vd[0]), V + DT * max(0.0, Vd[1])); Vp = (max(Vp[0], 1.0), max(Vp[1], 1.0))
    gd_now = irates((max(V, 1.0), V), (gam, gam), (al, al), (Tc, Tc), (Tr, Tr))[1]   # robust rate interval at the current state
    h3 = gam - PL.GAM_MIN + TAU * gd_now[0]; h4 = PL.GAM_MAX - gam - TAU * gd_now[1]
    ali_full = iadd(thi, ineg(gp)); sa_i = isin(ali_full); tc_sin_lo = min(tci[0] * sa_i[0], tci[0] * sa_i[1], tci[1] * sa_i[0], tci[1] * sa_i[1]); tc_sin_hi = max(tci[0] * sa_i[0], tci[0] * sa_i[1], tci[1] * sa_i[0], tci[1] * sa_i[1])
    cg_lo = min(np.cos(gp[0]), np.cos(gp[1]))
    qb_hi = 0.5 * RHO * Vp[1] ** 2
    # required bounds on gd(x+): flight-path barriers (look-ahead TAU) and sink/climb-rate barriers (GD_LIM)
    gd_req_lo = max((PL.GAM_MIN + (1 - eta) * h3 - gp[0]) / TAU, -GD_LIM + (1 - eta) * (gd_now[0] + GD_LIM))
    gd_req_hi = min((PL.GAM_MAX - (1 - eta) * h4 - gp[1]) / TAU, GD_LIM - (1 - eta) * (GD_LIM - gd_now[1]))
    den_lo = M * (Vp[1] if gd_req_lo >= 0 else Vp[0]); den_hi = M * (Vp[0] if gd_req_hi >= 0 else Vp[1])
    for CLa_w in (lo["CLa"], hi["CLa"]):
        for a, c in TH_C:
            for bb, d in TRc:
                # lower bound of the numerator: qb_lo S (CL0_lo + CLa_w (a th + c thsp - gp_hi)) + tc_sin_lo + (b Tr + d Trc) cA - m g cg  >= den_lo * gd_req_lo
                A.append([-qb * S * CLa_w * c, 0, -d * cA]); b.append(qb * S * (lo["CL0"] + CLa_w * (a * th - gp[1])) + tc_sin_lo + bb * Tr * cA - M * G * cg - den_lo * gd_req_lo)
                # upper bound of the numerator: qb_hi S (CL0_hi + CLa_w (a th + c thsp - gp_lo)) + tc_sin_hi + (b Tr + d Trc) - m g cg_lo <= den_hi * gd_req_hi
                A.append([qb_hi * S * CLa_w * c, 0, d]); b.append(den_hi * gd_req_hi - qb_hi * S * (hi["CL0"] + CLa_w * (a * th - gp[0])) - tc_sin_hi - bb * Tr + M * G * cg_lo)
    adm = admissible(u_prev)
    for i in range(3):
        e = [0.0, 0.0, 0.0]; e[i] = 1.0; A.append(e); b.append(adm[i][1]); A.append([-v for v in e]); b.append(-adm[i][0])
    A = np.array(A); b = np.array(b); n = np.linalg.norm(A, axis=1); n[n == 0] = 1.0
    return A / n[:, None], b / n
W = np.diag([1.0, 1.0 / PL.K_C, 1.0 / PL.K_R])
def project(u_nn, A, b):
    """returns (u, active, feasible)"""
    if np.all(A @ u_nn <= b + 1e-9): return u_nn.copy(), False, True
    lp = linprog(np.zeros(3), A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
    if lp.status != 0: return None, True, False
    r = minimize(lambda u: float(((W @ (u - u_nn)) ** 2).sum()), lp.x, jac=lambda u: 2 * W.T @ W @ (u - u_nn),
                 constraints=[dict(type="ineq", fun=lambda u: b - A @ u, jac=lambda u: -A)], method="SLSQP", options=dict(maxiter=200, ftol=1e-12))
    u = r.x if np.all(A @ r.x <= b + 1e-6) else lp.x
    return u, True, True
def filter_command(x, u_prev, u_nn, KL=PL.KL_ROB, eta=0.5, h_floor=3.0, gains=None):
    A, b = rows(x, u_prev, KL, eta, h_floor, gains); return project(u_nn, A, b)

# ---------------------------------------------------------------------------------------------------------------------
# set-membership contraction of the actuator gain intervals from the window (thrust states measured within +-EPS_T)
EPS_T = dict(c=0.4, r=1.0)      # measurement/estimation error bound on the thrust states [N]
def contract_gain(T_prev, Tc_prev, T_now, lam_int, beta_int, eps, nb=40):
    """interval hull of the gains beta consistent with the transitions T_now = b T_prev + (1-b) beta Tc_prev,
    b = exp(-lam DT) with lam in lam_int, |noise| <= eps. Arrays over the window. Returns (beta_lo, beta_hi)."""
    T_prev, Tc_prev, T_now = (np.asarray(v, float) for v in (T_prev, Tc_prev, T_now))
    bs = np.exp(-np.linspace(lam_int[1], lam_int[0], nb) * DT); lo_all, hi_all = [], []
    for b in bs:
        den = (1 - b) * Tc_prev; ok = den > 1e-6
        if not ok.any(): continue
        l = (T_now[ok] - eps - b * T_prev[ok]) / den[ok]; h = (T_now[ok] + eps - b * T_prev[ok]) / den[ok]
        l, h = max(l.max(), beta_int[0]), min(h.min(), beta_int[1])
        if l <= h: lo_all.append(l); hi_all.append(h)
    if not lo_all: return beta_int          # inconsistent data (should not happen): keep the box
    return (min(lo_all), max(hi_all))
def gain_corners(beta_c_int, beta_r_int):
    tc = [(np.exp(-l * DT), (1 - np.exp(-l * DT)) * b) for l in PL.BOX["lam_c"] for b in beta_c_int]
    tr = [(np.exp(-l * DT), (1 - np.exp(-l * DT)) * b) for l in PL.BOX["lam_r"] for b in beta_r_int]
    return tc, tr
