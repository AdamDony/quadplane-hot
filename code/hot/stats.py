"""Statistics over the campaign results: mean/std per controller and cell, Wilcoxon signed-rank against HOT, Friedman mean ranks."""
import os, sys, json, glob, numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
H = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(H, "..", "..", "results")
METRICS = ["t_target", "energy_Wh", "lf_min", "al_max_deg", "viol_pct"]; BETTER = dict(t_target="low", energy_Wh="low", lf_min="high", al_max_deg="low", viol_pct="low")
def load_all():
    rows = []
    for f in glob.glob(os.path.join(RES, "*.json")):
        if os.path.basename(f) in ("timing.json", "stats.json"): continue
        d = json.load(open(f))
        if isinstance(d, list): rows += [x for x in d if isinstance(x, dict) and "ctrl" in x]
    return rows
def table(rows, cell_prefix, ctrls):
    """per controller: mean/std of each metric over the rows of the cell (both directions pooled), success rate"""
    out = {}
    for c in ctrls:
        r = [x for x in rows if x["ctrl"] == c and x["cell"].startswith(cell_prefix)]
        if not r: continue
        d = dict(n=len(r), reached=float(np.mean([x["reached"] for x in r])))
        for m in METRICS:
            v = np.array([x[m] for x in r], float); v = v[np.isfinite(v)] if m == "t_target" else v
            d[m] = (float(np.mean(v)) if len(v) else float("nan"), float(np.std(v)) if len(v) else float("nan"))
        out[c] = d
    return out
def paired(rows, cell_prefix, ctrls, ref="hot"):
    """Wilcoxon signed-rank of each controller against the reference over matched (plant, direction, seed) keys; +/=/- counts per metric; Friedman ranks"""
    def key(x): return (x["cell"], x["rho0"], x["plant_id"], x["seed"])
    ref_rows = {key(x): x for x in rows if x["ctrl"] == ref and x["cell"].startswith(cell_prefix)}
    res = {}; ranks = {m: {c: [] for c in ctrls} for m in METRICS}
    for c in ctrls:
        if c == ref: continue
        cr = {key(x): x for x in rows if x["ctrl"] == c and x["cell"].startswith(cell_prefix)}
        common = [k for k in ref_rows if k in cr]
        if len(common) < 5: continue
        d = {}
        for m in METRICS:
            a = np.array([ref_rows[k][m] for k in common], float); b = np.array([cr[k][m] for k in common], float)
            pen = 20.0 if m == "t_target" else 1e3
            a = np.where(np.isfinite(a), a, pen); b = np.where(np.isfinite(b), b, pen)     # unreached counts as the horizon (20 s)
            diff = (b - a) if BETTER[m] == "low" else (a - b)                                # positive = reference better
            try: p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
            except ValueError: p = 1.0
            dm = np.median(diff) if np.median(diff) != 0 else diff.mean()
            d[m] = dict(p=float(p), ref_better=bool(p < 0.05 and dm > 0), other_better=bool(p < 0.05 and dm < 0), n=len(common))
        res[c] = d
    # Friedman ranks over the matched keys of all controllers
    keys = None
    present = [c for c in ctrls if any(x["ctrl"] == c and x["cell"].startswith(cell_prefix) for x in rows)]
    for c in present:
        cr = {key(x) for x in rows if x["ctrl"] == c and x["cell"].startswith(cell_prefix)}; keys = cr if keys is None else keys & cr
    fr = {}
    if keys and len(keys) >= 5:
        keys = sorted(keys)
        for m in METRICS:
            mat = np.array([[next(x[m] for x in rows if x["ctrl"] == c and key(x) == k) for c in present] for k in keys], float); mat = np.where(np.isfinite(mat), mat, 20.0 if m == "t_target" else 1e3)
            if BETTER[m] == "high": mat = -mat
            rk = np.array([rankdata(row) for row in mat]); fr[m] = dict(zip(present, rk.mean(0).round(2).tolist())); fr[m]["n"] = len(keys)
            try: fr[m]["p"] = float(friedmanchisquare(*[mat[:, i] for i in range(len(present))]).pvalue)
            except Exception: fr[m]["p"] = float("nan")
    return res, fr
if __name__ == "__main__":
    rows = load_all(); ctrls = sorted({x["ctrl"] for x in rows}); print("controllers", ctrls, "rows", len(rows))
    for cell in ("nominal", "cross", "outbox", "gust", "noise"):
        t = table(rows, cell, ctrls)
        if t: print(cell, json.dumps({c: {k: (round(v[0], 2), round(v[1], 2)) if isinstance(v, tuple) else round(v, 2) for k, v in d.items()} for c, d in t.items()}, indent=0)[:1500])
    json.dump(dict(tables={cell: table(rows, cell, ctrls) for cell in ("nominal", "cross", "outbox", "gust", "noise")}, paired={cell: paired(rows, cell, ctrls) for cell in ("cross", "outbox", "gust", "noise")}), open(os.path.join(RES, "stats.json"), "w"), indent=1)
