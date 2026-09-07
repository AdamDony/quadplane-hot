"""Generate figsrc/fig12_ablation.tex (grouped bars, EmT style) from results/variant_*.json."""
import os, sys, json, glob, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(H, "..", ".."); RES = os.path.join(ROOT, "results"); FS = os.path.join(ROOT, "figsrc")
V = {}
for f in glob.glob(os.path.join(RES, "variant_*.json")):
    for x in json.load(open(f)): V.setdefault(x["ctrl"], []).append(x)
def m(tag, key, filt="_nf"):
    r = V.get(tag + filt, []); v = [x[key] for x in r]; v = [q for q in v if np.isfinite(q)]; return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)
groups = [("modules removed", [("hot_v4", "HOT"), ("hot_nophys", "$-$phys."), ("hot_noaux", "$-$aux."), ("lstm_v2", "LSTM"), ("mlp_v2", "MLP")]),
          ("window $L$", [("hot_L8", "8"), ("hot_L16", "16"), ("hot_v4", "32"), ("hot_L64", "64")]),
          ("width $d$", [("hot_d32", "32"), ("hot_v4", "48"), ("hot_d96", "96")]),
          ("depth $N$", [("hot_b1", "1"), ("hot_v4", "2"), ("hot_b4", "4")])]
s = ["\\documentclass[tikz,border=2pt]{standalone}", "\\input{preamble}", "\\definecolor{chot}{HTML}{C0392B}", "\\begin{document}", "\\begin{tikzpicture}",
     "\\begin{groupplot}[group style={group size=4 by 1, horizontal sep=11mm}, paperaxis, width=3.2cm, height=2.6cm, ybar, bar width=4pt, tick label style={font=\\tiny}, x tick label style={font=\\scriptsize}, enlarge x limits=0.2, ymin=0, error bars/y dir=both, error bars/y explicit, error bars/error mark=none]"]
for gi, (title, items) in enumerate(groups):
    present = [(t, l) for t, l in items if t + "_nf" in V]
    if not present: continue
    labels = ",".join(l for _, l in present)
    s.append("\\nextgroupplot[title={%s}, xtick={%s}, xticklabels={%s}, ylabel={%s}, axis y line*=left]" % (title, ",".join(str(i) for i in range(len(present))), labels, "time to target [s]" if gi == 0 else ""))
    s.append("\\addplot[fill=chot!30, draw=chot] coordinates {%s};" % " ".join("(%d,%.3f) +- (0,%.3f)" % (i, *m(t, "t_target")) for i, (t, _) in enumerate(present)))
    s.append("\\panel{%s}{}" % "abcd"[gi])
s += ["\\end{groupplot}", "\\end{tikzpicture}", "\\end{document}"]
# second row: violation rate of the bare policy
s2 = s[:-3]
open(os.path.join(FS, "fig12_ablation.tex"), "w").write("\n".join(s)); print("written fig12_ablation.tex with", {k: len(v) for k, v in V.items()})
