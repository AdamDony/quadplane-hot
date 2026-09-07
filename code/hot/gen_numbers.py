"""Write paper/numbers.tex from configs, the safety-module constants and the results (what exists is written; the rest stays '?')."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hot import plant as PL, bsf
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(H, "..", ".."); RUNS = os.path.join(H, "..", "runs"); RES = os.path.join(ROOT, "results"); DES = os.path.join(ROOT, "design")
M = {}
def load(p): return json.load(open(p)) if os.path.exists(p) else None
cfg = load(os.path.join(RUNS, "hot_v4_config.json")) or {}
M.update(numCross="100", numAR="%.2f" % (2.0 / 0.292), numKL="%.2f" % PL.KL_ROB, numGdLim="%.1f" % bsf.GD_LIM, numL=str(cfg.get("L", 32)), numD=str(cfg.get("d", 48)), numBlocks=str(cfg.get("blocks", 2)), numHeads=str(cfg.get("heads", 4)),
         numParams=("{:,}".format(cfg["params"]) if "params" in cfg else "?"), numIters=str(cfg.get("iters", 800)), numBatch=str(cfg.get("batch", 32)), numLr="$2\\times10^{-4}$", numWE="0.2", numWB="2", numWA="1", numWS="0.3",
         numPlantsOnline="128", numSamplesOnline="32", numKgam="%.1f" % bsf.K_GD, numKr="%.0f" % bsf.K_R, numRlo="%.2f" % bsf.GD_REF[0], numRhi="%.2f" % bsf.GD_REF[1], numVblo="%.0f" % bsf.V_BAND[0], numVbhi="%.1f" % bsf.V_BAND[1], numH=str(bsf.H_BACKUP), numTauB="%.2f" % bsf.TAU_LEAD, numKc="%.0f" % bsf.K_C_B, numKv="%.1f" % bsf.K_V_B, numAmax="%.1f" % bsf.A_MAX_B, numCvd="%.1f" % bsf.TERM["vd_abs"],
         numMargins="%.2f,\\ %.0f^\\circ,\\ %.0f^\\circ,\\ %.0f^\\circ,\\ %.0f^\\circ,\\ %.0f^\\circ,\\ %.2f,\\ %.1f" % (bsf.MARGIN[0], np.degrees(bsf.MARGIN[1]), np.degrees(bsf.MARGIN[2]), np.degrees(bsf.MARGIN[3]), np.degrees(bsf.MARGIN[4]), np.degrees(bsf.MARGIN[5]), bsf.MARGIN[6], bsf.MARGIN[7]),
         numClfLo="%.2f" % bsf.TERM["lift"][0], numClfHi="%.2f" % bsf.TERM["lift"][1], numCgam="%.0f" % np.degrees(bsf.TERM["gam_abs"]), numCgd="%.2f" % bsf.TERM["gd_abs"], numCalLo="%.0f" % np.degrees(bsf.TERM["al"][0]), numCalHi="%.1f" % np.degrees(bsf.TERM["al"][1]), numCvLo="%.0f" % bsf.TERM["V"][0], numCvHi="%.0f" % bsf.TERM["V"][1],
         numEps="%.1f\\ \\mathrm{m/s},\\ %.0f^\\circ,\\ %.0f^\\circ,\\ %.1f\\ \\mathrm N,\\ %.0f\\ \\mathrm N" % (bsf.EPS_CONS[0], np.degrees(bsf.EPS_CONS[1]), np.degrees(bsf.EPS_CONS[2]), bsf.EPS_CONS[3], bsf.EPS_CONS[4]), numPlantsFull="1088")
g = bsf.K_R * PL.BOX["beta_r"][1] * (1 - np.exp(-PL.BOX["lam_r"][1] * PL.DT)) / (PL.M * 4.0); M["numKrGain"] = "%.2f" % g
M["numKcGain"] = "%.2f" % (bsf.K_C_B * PL.BOX["beta_c"][1] * (1 - np.exp(-PL.BOX["lam_c"][1] * PL.DT)) / PL.M)
st = load(os.path.join(DES, "settle_numbers.json"))
if st: M.update(numSettlePairs="{:,}".format(st["pairs"]), numSettleFail=str(st["failures"]), numSettleWorst="%.3f" % (st["margin_lift"] + st["worst_margin_rel"]))
# results
import glob as _glob
def rows_all():
    rows = []
    for f in _glob.glob(os.path.join(RES, "*.json")):
        if os.path.basename(f) in ("timing.json", "stats.json", "interp.json"): continue
        d = json.load(open(f))
        if isinstance(d, list): rows += [x for x in d if isinstance(x, dict) and "ctrl" in x]
    return rows
ROWS = rows_all(); ST = load(os.path.join(RES, "stats.json")) or {}
def nom(ctrl, rho0, key, d=1):
    r = [x for x in ROWS if x["ctrl"] == ctrl and x["cell"] == "nominal" and x["rho0"] == rho0]
    if not r: return "?"
    v = r[0].get(key); return "?" if v is None or (isinstance(v, float) and not np.isfinite(v)) else ("%.*f" % (d, v) if isinstance(v, float) else str(v))
for c, tag in (("hot", "hot"), ("threshold", "thr"), ("schedule", "sch"), ("gspid", "pid"), ("mpc", "mpc"), ("ppo", "ppo"), ("sac", "sac"), ("mlp", "mlp"), ("lstm", "lstm"), ("hot_nofilter", "hotnf")):
    M[tag + "FrontT"] = nom(c, 6.0, "t_target"); M[tag + "BackT"] = nom(c, 15.0, "t_target"); M[tag + "FrontLf"] = nom(c, 6.0, "lf_min", 2); M[tag + "BackLf"] = nom(c, 15.0, "lf_min", 2)
    M[tag + "FrontAl"] = nom(c, 6.0, "al_max_deg"); M[tag + "BackAl"] = nom(c, 15.0, "al_max_deg"); M[tag + "FrontEnergy"] = nom(c, 6.0, "energy_Wh", 2); M[tag + "BackEnergy"] = nom(c, 15.0, "energy_Wh", 2)
    M[tag + "FrontInterv"] = nom(c, 6.0, "interventions"); M[tag + "BackInterv"] = nom(c, 15.0, "interventions"); M[tag + "FrontUpd"] = "400"; M[tag + "BackUpd"] = "400"
    T = ST.get("tables", {})
    for cell, ctag in (("cross", "Cross"), ("outbox", "Out"), ("gust", "Gust"), ("noise", "Noise")):
        d = T.get(cell, {}).get(c)
        if d:
            M[tag + ctag + "Reached"] = "%.0f\%%" % (100 * d["reached"]); M[tag + ctag + "T"] = "%.1f" % d["t_target"][0]; M[tag + ctag + "Lf"] = "%.2f" % d["lf_min"][0]; M[tag + ctag + "Viol"] = "%.1f" % d["viol_pct"][0]
            rr = [x for x in ROWS if x["ctrl"] == c and x["cell"].startswith(cell) and "interventions" in x]
            if rr: M[tag + ctag + "Interv"] = "%.0f" % (100 * np.mean([x["interventions"] / 400.0 for x in rr]))
    fr = ST.get("paired", {}).get("cross", [{}, {}])[1].get("t_target", {}) if ST else {}
    if c in fr: M[tag + "CrossRank"] = "%.2f" % fr[c]
fr = ST.get("paired", {}).get("cross", [{}, {}])[1].get("t_target", {}) if ST else {}
others = {k: v for k, v in fr.items() if k not in ("hot", "hot_nofilter", "p", "n")}
M["hotnfCrossRank"] = "%.2f" % fr["hot_nofilter"] if "hot_nofilter" in fr else "?"
M["bestOtherCrossName"] = {"threshold": "the threshold rule", "schedule": "the schedule", "gspid": "the gain-scheduled PID", "mpc": "the MPC", "ppo": "PPO", "sac": "SAC"}.get(min(others, key=others.get), "?") if others else "?"
M["numFriedmanN"] = str(fr.get("n", "?"))
vio = [ST["tables"]["cross"][c]["viol_pct"][0] for c in ("mlp", "lstm", "ppo", "sac", "hot_nofilter") if ST and c in ST.get("tables", {}).get("cross", {})]
if vio: M["maxOtherCrossViol"] = "%.1f" % max(vio)
if others: M["bestOtherCrossRank"] = "%.2f" % min(others.values())
M["numOutbox"] = "50"
def agg(name, cell, rho):
    p = glob.glob(os.path.join(RES, f"{name}_*.json")); rows = []
    for f in p: rows += [r for r in json.load(open(f)) if r["cell"] == cell and r["rho0"] == rho[0]]
    return rows
r = agg("hot", "nominal", (6.0, 15.0))
if r: M.update(hotFrontT="%.1f" % r[0]["t_target"], hotFrontLf="%.2f" % r[0]["lf_min"])
tm = load(os.path.join(RES, "timing.json"))
if tm: M.update(hotEvalMs="%.1f" % (tm["hot_eval_us"] / 1e3), hotCheckMs="%.0f" % tm["check_ms_mean"], hotCheckMaxMs="%.0f" % tm["check_ms_max"], numParams="{:,}".format(tm["params"]), thrUs="%.0f" % tm["threshold_us"], schUs="%.0f" % tm["schedule_us"], pidUs="%.0f" % tm["gspid_us"])
r = [x for x in ROWS if x["ctrl"] == "mpc" and x["cell"] == "nominal"]
if r: M["mpcStepMs"] = "%.0f" % np.mean([x["wall_ms_per_step"] for x in r])
# replay, firmware in the loop, interpretability, training, ablations
for name, tag in (("front", "Front"), ("back", "Back")):
    r = [x for x in ROWS if x["ctrl"] == "hot" and x["cell"] == f"replay_{name}"]
    if r: M[f"hotReplay{tag}T"] = "%.1f" % r[0]["t_target"] if np.isfinite(r[0]["t_target"]) else "?"; M[f"hotReplay{tag}Lf"] = "%.2f" % r[0]["lf_min"]; M[f"hotReplay{tag}Al"] = "%.1f" % r[0]["al_max_deg"]; M[f"hotReplay{tag}Interv"] = str(r[0].get("interventions", "?"))
sil = load(os.path.join(RES, "sil_hot.json"))
if sil:
    for name, tag in (("front", "Front"), ("back", "Back")):
        if name in sil:
            d = sil[name]; M[f"sil{tag}T"] = "%.1f" % d["t_target"] if d["t_target"] else "?"; M[f"sil{tag}AlMin"] = "%.1f" % d["al_min_deg"]; M[f"sil{tag}AlMax"] = "%.1f" % d["al_max_deg"]; M[f"sil{tag}Gam"] = "%.1f" % d["gam_abs_max_deg"]; M[f"sil{tag}Lf"] = "%.2f" % d["lf_min"]; M[f"sil{tag}Interv"] = str(d["interventions"]); M[f"sil{tag}Upd"] = str(d["n_upd"])
    M["silUpdMs"] = "%.0f" % np.mean([sil[n]["upd_ms"] for n in sil])
ip = load(os.path.join(RES, "interp.json"))
if ip and "best" in ip:
    b = ip["best"]
    for k, nm in (("CL0", "CLzero"), ("CLa", "CLa"), ("kappa", "Kappa"), ("beta_r", "BetaR"), ("CD0", "CDzero"), ("k", "Kdrag"), ("g", "Gth"), ("lam_r", "LamR")):
        M["interpRtwo" + nm] = "%.2f" % b[k]["r2"]; M["interpT" + nm] = "%.1f" % b[k]["t"]; M["interpTr" + nm] = b[k]["transition"]; M["interpShrink" + nm] = "%.2f" % b[k]["shrink"]
    M["interpRtwoMean"] = "%.2f" % np.mean([b[k]["r2"] for k in b]); M["interpAttMax"] = "%.3f" % ip["att_max"]; M["interpAttMin"] = "%.3f" % ip["att_min"]; M["interpAttUniform"] = "%.3f" % ip["att_uniform"]
hist = load(os.path.join(RUNS, "hot_v4_hist.json"))
if hist:
    b = [h["barrier"] for h in hist]; M.update(trainBarrierDrop="a factor of %.0f" % (np.mean(b[:20]) / max(np.mean(b[-20:]), 1e-6)), trainReached="%.2f" % np.mean([h["reached"] for h in hist[-20:]]), trainAux="%.2f" % np.mean([h["aux"] for h in hist[-20:]]), trainWallMin="%.0f" % (hist[-1]["wall"] / 60))
abl = {}
for f in _glob.glob(os.path.join(RES, "variant_*.json")):
    for x in json.load(open(f)): abl.setdefault(x["ctrl"], []).append(x)
def ab(tag, key, d=1, filt="_nf"):
    r = abl.get(tag + filt)
    if not r: return "?"
    v = [x[key] for x in r]; v = [q for q in v if np.isfinite(q)]; return "%.*f" % (d, np.mean(v)) if v else "?"
M.update(numAblPlants="30", ablNoPhysT=ab("hot_nophys", "t_target"), ablNoPhysViol=ab("hot_nophys", "viol_pct"), ablNoAuxT=ab("hot_noaux", "t_target"), ablBestL="32")
def abr(tag, filt):
    r = abl.get(tag + filt); return "%.0f" % (100 * np.mean([x["reached"] for x in r])) if r else "?"
def abt(tag, filt):
    r = abl.get(tag + filt); v = [x["t_target"] for x in r if np.isfinite(x["t_target"])] if r else []; return "%.1f" % np.median(v) if v else "?"
for tag, nm in (("hot_v4", "Full"), ("hot_nophys", "NoPhys"), ("hot_noaux", "NoAux"), ("hot_L8", "Lsmall"), ("hot_d32", "Dsmall"), ("hot_b1", "Bone"), ("mlp_v2", "Mlp"), ("lstm_v2", "Lstm")):
    M["abl" + nm + "Reached"] = abr(tag, "_nf"); M["abl" + nm + "ReachedF"] = abr(tag, "_f"); M["abl" + nm + "Tmed"] = abt(tag, "_nf"); M["abl" + nm + "Viol"] = ab(tag, "viol_pct", 1); M["abl" + nm + "Lf"] = ab(tag, "lf_min", 2)
ipn = load(os.path.join(RES, "interp_noaux.json"))
if ipn: M["ablNoAuxRtwo"] = "%.2f" % np.mean([v["r2"] for v in ipn["best"].values()]) if "best" in ipn else "%.2f" % np.mean(list(ipn["r2"].values()))
out = os.path.join(ROOT, "paper", "numbers.tex")

def _sub(cell, rho0):
    return [x for x in ROWS if x["ctrl"] == "hot" and x["cell"] == cell and x["rho0"] == rho0]
for cell, ctag in (("cross", "Cross"), ("outbox", "Out")):
    for rho0, dtag in ((6.0, "Front"), (15.0, "Back")):
        rr = _sub(cell, rho0)
        if rr:
            M["hot" + ctag + dtag + "Reached"] = "%d" % sum(x["reached"] for x in rr); M["hot" + ctag + dtag + "N"] = str(len(rr)); M["hot" + ctag + dtag + "Tmed"] = "%.1f" % np.nanmedian([x["t_target"] for x in rr])
            M["hot" + ctag + dtag + "Interv"] = "%.1f" % np.mean([x["interventions"] for x in rr]); M["hot" + ctag + dtag + "Unver"] = "%.1f" % np.mean([x["unverified"] for x in rr])
            M["hot" + ctag + dtag + "IntervPct"] = "%.1f" % (100 * np.mean([x["interventions"] for x in rr]) / 400.0)
            M["hot" + ctag + dtag + "StalledUnver"] = "%d" % sum(1 for x in rr if not x["reached"] and x["unverified"] > 0); M["hot" + ctag + dtag + "StalledNoInt"] = "%d" % sum(1 for x in rr if not x["reached"] and x["interventions"] == 0)
for lvl, tag in (("0.25", "Qtr"), ("0.5", "Half"), ("1.0", "One"), ("1.5", "OneHalf")):
    rr = [x for x in ROWS if x["ctrl"] == "hot" and x["cell"] == "gust" + lvl]
    if rr: M["hotGust" + tag + "Viol"] = "%.1f" % np.mean([x["viol_pct"] for x in rr]); M["hotGust" + tag + "Lf"] = "%.2f" % min(x["lf_min"] for x in rr); M["hotGust" + tag + "Interv"] = "%.0f" % (100 * np.mean([x["interventions"] for x in rr]) / 400.0); M["hotGust" + tag + "Reached"] = "%d" % sum(x["reached"] for x in rr); M["hotGust" + tag + "Al"] = "%.1f" % max(x["al_max_deg"] for x in rr)
    for c, ct in (("gspid", "pid"), ("threshold", "thr"), ("schedule", "sch"), ("sac", "sac")):
        r2 = [x for x in ROWS if x["ctrl"] == c and x["cell"] == "gust" + lvl]
        if r2: M[ct + "Gust" + tag + "Viol"] = "%.1f" % np.mean([x["viol_pct"] for x in r2]); M[ct + "Gust" + tag + "Lf"] = "%.2f" % min(x["lf_min"] for x in r2)
for lvl, tag in (("0.5", "Half"), ("1.0", "One"), ("2.0", "Two")):
    for rho0, dtag in ((6.0, "Front"), (15.0, "Back")):
        rr = [x for x in ROWS if x["ctrl"] == "hot" and x["cell"] == "noise" + lvl and x["rho0"] == rho0]
        if rr: M["hotNoise" + tag + dtag + "Reached"] = "%d" % sum(x["reached"] for x in rr); M["hotNoise" + tag + dtag + "Tmed"] = "%.1f" % np.nanmedian([x["t_target"] for x in rr])
rr = [x for x in ROWS if x["ctrl"] == "hot" and x["cell"] == "gust1.0"] + [x for x in ROWS if x["ctrl"] == "hot" and x["cell"] == "gust1.5"]
if rr: M["hotGustStrongViol"] = "%.0f" % np.mean([x["viol_pct"] for x in rr]); M["hotGustStrongInterv"] = "%.0f" % (100 * np.mean([x["interventions"] for x in rr]) / 400.0); M["hotGustStrongUnverShare"] = "%.0f" % (100 * np.mean([x["unverified"] for x in rr]) / max(np.mean([x["interventions"] for x in rr]), 1e-9))
rr = [x for x in ROWS if x["ctrl"] == "hot" and x["cell"] == "replay_back"]
if rr: M["hotReplayBackUnver"] = str(rr[0]["unverified"])

import re as _re
used = set()
for f in _glob.glob(os.path.join(ROOT, "paper", "*.tex")):
    if f.endswith("numbers.tex"): continue
    used |= set(_re.findall(r"\\((?:num|hot|hotnf|thr|sch|pid|mpc|ppo|sac|mlp|lstm|bestOther|maxOther|sil|abl|interp)[A-Z][A-Za-z]*)", open(f).read()))
for k in used:
    if k not in M: M[k] = "?"
existing = {}
if os.path.exists(out):
    import re
    for m in re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}', open(out).read()): existing[m[0]] = m[1]
for k in existing:
    if k not in M: M[k] = existing[k]
with open(out, "w") as fh:
    fh.write("%% generated by code/hot/numbers.py\n")
    for k in sorted(M): fh.write("\\newcommand{\\%s}{%s}\n" % (k, M[k]))
print("numbers written:", len(M), "unknown:", [k for k, v in M.items() if v == "?"])
