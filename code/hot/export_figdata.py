"""Export figure data (.dat for pgfplots) from the training histories, campaign results and trajectories."""
import os, sys, json, glob, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(H, "..", ".."); RUNS = os.path.join(H, "..", "runs"); RES = os.path.join(ROOT, "results"); FD = os.path.join(ROOT, "figdata"); os.makedirs(FD, exist_ok=True)
def smooth(v, w=10): v = np.asarray(v, float); return np.convolve(np.pad(v, (w - 1, 0), mode="edge"), np.ones(w) / w, mode="valid")
# training curves
for tag in ("hot_v4", "mlp_v2", "lstm_v2", "hot_nophys", "hot_noaux"):
    f = os.path.join(RUNS, f"{tag}_hist.json")
    if not os.path.exists(f): continue
    h = json.load(open(f))
    with open(os.path.join(FD, f"train_{tag}.dat"), "w") as fh:
        fh.write("it n time energy barrier aux reached lf_min al_max\n")
        for i, r in enumerate(h): fh.write("%d %d %.4f %.4f %.4f %.4f %.3f %.3f %.2f\n" % (r["it"], r["n"], r["time"], r["energy"], smooth([q["barrier"] for q in h])[i], r["aux"], smooth([q["reached"] for q in h])[i], r["lf_min"], r["al_max_deg"]))
# nominal trajectories
for f in glob.glob(os.path.join(RES, "traj_*.npz")):
    d = np.load(f); X, U, LF = d["X"], d["U"], d["LF"]; name = os.path.basename(f)[5:-4]
    with open(os.path.join(FD, f"traj_{name}.dat"), "w") as fh:
        fh.write("t V gam alpha th Tc Tr thsp Tcc Trc lf gd\n")
        for k in range(0, len(X), 2): fh.write("%.2f %.3f %.3f %.3f %.3f %.2f %.2f %.3f %.2f %.2f %.3f %.3f\n" % ((k + 1) * 0.05, X[k, 0], np.degrees(X[k, 1]), np.degrees(X[k, 2] - X[k, 1]), np.degrees(X[k, 2]), X[k, 3], X[k, 4], np.degrees(U[k, 0]), U[k, 1], U[k, 2], LF[k], d["GD"][k]))
print("figdata exported:", len(os.listdir(FD)), "files")
# ---- campaign results: cross-plant box statistics, gust and noise sweeps, outbox ----
rows = []
for f in glob.glob(os.path.join(RES, "*.json")):
    if os.path.basename(f) in ("timing.json", "stats.json"): continue
    d = json.load(open(f))
    if isinstance(d, list): rows += [x for x in d if isinstance(x, dict) and "ctrl" in x]
ctrls = ["threshold", "schedule", "gspid", "mpc", "mlp", "lstm", "hot_nofilter", "ppo", "sac", "hot"]
metrics = ["t_target", "energy_Wh", "lf_min", "al_max_deg", "viol_pct"]
def box(v):
    v = np.array(v, float); v = v[np.isfinite(v)]
    return (np.min(v), np.percentile(v, 25), np.median(v), np.percentile(v, 75), np.max(v)) if len(v) else (np.nan,) * 5
for cell in ("cross", "outbox"):
    with open(os.path.join(FD, f"box_{cell}.dat"), "w") as fh:
        fh.write("idx ctrl metric lw q1 med q3 uw n reached\n")
        for m in metrics:
            for i, c in enumerate(ctrls):
                r = [x for x in rows if x["ctrl"] == c and x["cell"] == cell]
                if not r: continue
                b = box([x[m] for x in r]); fh.write("%d %s %s %s %d %.3f\n" % (i + 1, c, m, " ".join("%.4f" % q for q in b), len(r), np.mean([x["reached"] for x in r])))
for cell in ("gust", "noise"):
    with open(os.path.join(FD, f"sweep_{cell}.dat"), "w") as fh:
        fh.write("ctrl level t_mean t_std lf_mean lf_std viol_mean reached\n")
        for c in ctrls:
            for lev in ((0.25, 0.5, 1.0, 1.5) if cell == "gust" else (0.5, 1.0, 2.0)):
                r = [x for x in rows if x["ctrl"] == c and x["cell"] == f"{cell}{lev}"]
                if not r: continue
                t = np.array([x["t_target"] for x in r], float); t = t[np.isfinite(t)]
                fh.write("%s %.1f %.3f %.3f %.4f %.4f %.3f %.3f\n" % (c, lev, t.mean() if len(t) else np.nan, t.std() if len(t) else np.nan, np.mean([x["lf_min"] for x in r]), np.std([x["lf_min"] for x in r]), np.mean([x["viol_pct"] for x in r]), np.mean([x["reached"] for x in r])))
print("campaign figdata exported")
