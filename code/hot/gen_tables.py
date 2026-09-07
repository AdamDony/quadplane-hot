"""LaTeX tables from results/stats.json: nominal metrics, cross-plant summary with Wilcoxon and Friedman, robustness summary."""
import os, sys, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(H, "..", ".."); RES = os.path.join(ROOT, "results"); TAB = os.path.join(ROOT, "paper", "tables"); os.makedirs(TAB, exist_ok=True)
S = json.load(open(os.path.join(RES, "stats.json")))
ORDER = ["threshold", "schedule", "gspid", "mpc", "mlp", "lstm", "hot_nofilter", "ppo", "sac", "hot"]; LAB = dict(threshold="THR", schedule="SCH", gspid="GS-PID", mpc="MPC", mlp="MLP", lstm="LSTM", hot_nofilter="HOT$-$LSM", ppo="PPO", sac="SAC", hot="\\textbf{HOT}")
def f(v, d=1): return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else ("%.*f" % (d, v))
def ms(t, d=1): return "%s$\\pm$%s" % (f(t[0], d), f(t[1], d))
def nominal():
    rows = json.load(open(os.path.join(RES, "all_rows_cache.json"))) if os.path.exists(os.path.join(RES, "all_rows_cache.json")) else None
    import glob
    rows = []
    for fn in glob.glob(os.path.join(RES, "*.json")):
        if os.path.basename(fn) in ("timing.json", "stats.json", "interp.json"): continue
        d = json.load(open(fn))
        if isinstance(d, list): rows += [r for r in d if isinstance(r, dict) and r.get("cell") == "nominal"]
    L = ["\\begin{tabular}{l|cccc|cccc}\\toprule", "& \\multicolumn{4}{c|}{front transition ($6\\to15$ m/s)} & \\multicolumn{4}{c}{back transition ($15\\to6$ m/s)}\\\\", "controller & time [s] & energy [Wh] & min $\\lf$ & max $\\alpha$ [deg] & time [s] & energy [Wh] & min $\\lf$ & max $\\alpha$ [deg]\\\\\\midrule"]
    for c in ORDER:
        r6 = [r for r in rows if r["ctrl"] == c and r["rho0"] == 6.0]; r15 = [r for r in rows if r["ctrl"] == c and r["rho0"] == 15.0]
        if not r6 or not r15: continue
        a, b = r6[0], r15[0]
        L.append("%s & %s & %s & %s & %s & %s & %s & %s & %s\\\\" % (LAB[c], f(a["t_target"]), f(a["energy_Wh"], 2), f(a["lf_min"], 2), f(a["al_max_deg"]), f(b["t_target"]), f(b["energy_Wh"], 2), f(b["lf_min"], 2), f(b["al_max_deg"])))
    L += ["\\bottomrule\\end{tabular}"]; open(os.path.join(TAB, "nominal.tex"), "w").write("\n".join(L))
def cross(cell="cross", fname="cross.tex"):
    T = S["tables"].get(cell, {}); P = S["paired"].get(cell, [{}, {}]); W, FR = P[0], P[1]
    L = ["\\begin{tabular}{l|c|cc|cc|cc|c|c}\\toprule", "controller & reached & time [s] & W & energy [Wh] & W & min $\\lf$ & W & viol.\\ [\\%] & rank\\\\\\midrule"]
    def wsym(c, m):
        if c == "hot" or c not in W or m not in W[c]: return ""
        d = W[c][m]; return "$+$" if d["ref_better"] else ("$-$" if d["other_better"] else "$=$")
    for c in ORDER:
        if c not in T: continue
        d = T[c]; rk = FR.get("t_target", {}).get(c, float("nan"))
        L.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s\\\\" % (LAB[c], f(d["reached"], 2), ms(d["t_target"]), wsym(c, "t_target"), ms(d["energy_Wh"], 2), wsym(c, "energy_Wh"), ms(d["lf_min"], 2), wsym(c, "lf_min"), f(d["viol_pct"][0]), f(rk, 2)))
    L += ["\\bottomrule\\end{tabular}"]; open(os.path.join(TAB, fname), "w").write("\n".join(L))
def robust():
    L = ["\\begin{tabular}{l|ccc|ccc|ccc}\\toprule", "& \\multicolumn{3}{c|}{outside the box} & \\multicolumn{3}{c|}{gusts} & \\multicolumn{3}{c}{noise}\\\\", "controller & reached & min $\\lf$ & viol.\\ [\\%] & reached & min $\\lf$ & viol.\\ [\\%] & reached & min $\\lf$ & viol.\\ [\\%]\\\\\\midrule"]
    for c in ORDER:
        cells = [S["tables"].get(k, {}).get(c) for k in ("outbox", "gust", "noise")]
        if all(v is None for v in cells): continue
        L.append("%s & %s\\\\" % (LAB[c], " & ".join(("%s & %s & %s" % (f(v["reached"], 2), ms(v["lf_min"], 2), f(v["viol_pct"][0], 1))) if v else "-- & -- & --" for v in cells)))
    L += ["\\bottomrule\\end{tabular}"]; open(os.path.join(TAB, "robust.tex"), "w").write("\n".join(L))
nominal(); cross(); cross("outbox", "outbox.tex"); robust(); print("tables written:", os.listdir(TAB))
