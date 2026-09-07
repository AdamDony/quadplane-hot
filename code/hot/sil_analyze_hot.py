"""Reduce the HOT software-in-the-loop logs to figure data and summary numbers."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hot import plant as PL
H = os.path.dirname(os.path.abspath(__file__)); SIL = os.path.join(H, "..", "sil"); FD = os.path.join(H, "..", "..", "figdata"); RES = os.path.join(H, "..", "..", "results")
nom = PL.nominal(1); out = {}
for mode in ("front", "back"):
    f = os.path.join(SIL, f"sil_{mode}.npz")
    if not os.path.exists(f): continue
    d = np.load(f); t, x, u, xh = d["t"], d["x"], d["u"], d["xh"]; alt = d["alt"]; interv = d["interv"] if "interv" in d.files else None
    V, gam, th = x.T; al = th - gam; X = np.column_stack([V, gam, th, xh[:, 3], xh[:, 4]]); lf = PL.lift_fraction(X, {k: np.full(len(V), PL.NOM[k]) for k in PL.KEYS})
    js = json.load(open(os.path.join(SIL, f"sil_{mode}.json")))
    with open(os.path.join(FD, f"sil_hot_{mode}.dat"), "w") as fh:
        fh.write("t V gam alpha th thsp Tcc Trc lf alt\n")
        for i in range(0, len(t), 2): fh.write("%.2f %.3f %.3f %.3f %.3f %.3f %.2f %.2f %.4f %.2f\n" % (t[i], V[i], np.degrees(gam[i]), np.degrees(al[i]), np.degrees(th[i]), np.degrees(u[i, 0]), u[i, 1], u[i, 2], lf[i], alt[i]))
    out[mode] = dict(t_target=js.get("t_done"), al_min_deg=float(np.degrees(al.min())), al_max_deg=float(np.degrees(al.max())), gam_abs_max_deg=float(np.degrees(np.abs(gam).max())), lf_min=float(lf.min()), n_upd=js["n_upd"], upd_ms=js["upd_ms_mean"], interventions=int(js["stats"].get("interventions", 0)), unverified=int(js["stats"].get("unverified_backup", 0)))
json.dump(out, open(os.path.join(RES, "sil_hot.json"), "w"), indent=1); print(out)
