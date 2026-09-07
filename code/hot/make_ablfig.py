"""Generate figsrc/fig12_ablation.tex: fraction of the 30 unseen plants reached within 20 s (bare policy and with the module)
and violation rate of the bare policy, for the modules removed, the baseline architectures and the reduced sizes."""
import os, json, glob, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(H, "..", ".."); RES = os.path.join(ROOT, "results"); FS = os.path.join(ROOT, "figsrc")
V = {}
for f in glob.glob(os.path.join(RES, "variant_*.json")):
    for x in json.load(open(f)): V.setdefault(x["ctrl"], []).append(x)
def reached(tag, filt): r = V.get(tag + filt, []); return 100 * np.mean([x["reached"] for x in r]) if r else np.nan
def viol(tag): r = V.get(tag + "_nf", []); return np.mean([x["viol_pct"] for x in r]) if r else np.nan
groups = [("modules removed, baselines", [("hot_v4", "HOT"), ("hot_nophys", "$-$phys."), ("hot_noaux", "$-$aux."), ("lstm_v2", "LSTM"), ("mlp_v2", "MLP")]),
          ("reduced sizes", [("hot_v4", "full"), ("hot_L8", "$L{=}8$"), ("hot_d32", "$d{=}32$"), ("hot_b1", "$N{=}1$")])]
s = ["\\documentclass[tikz,border=2pt]{standalone}", "\\input{preamble}", "\\definecolor{chot}{HTML}{C0392B}", "\\begin{document}", "\\begin{tikzpicture}",
     "\\begin{groupplot}[group style={group size=3 by 1, horizontal sep=12mm}, paperaxis, width=4.6cm, height=3.0cm, ybar, /pgf/bar width=4.5pt, tick label style={font=\\tiny}, x tick label style={font=\\scriptsize, rotate=25, anchor=north east}, label style={font=\\scriptsize}, enlarge x limits=0.18, ymin=0]"]
present = [(t, l) for t, l in groups[0][1] if t + "_nf" in V]
s.append("\\nextgroupplot[title={%s}, xtick={%s}, xticklabels={%s}, ylabel={reached within 20 s [\\%%]}, ymax=105, legend to name=ablleg, legend columns=2, legend style={font=\\scriptsize, draw=none, fill=none, /tikz/every even column/.append style={column sep=4pt}}, legend image code/.code={\\draw[#1] (0cm,-0.08cm) rectangle (0.25cm,0.08cm);}]" % (groups[0][0], ",".join(str(i) for i in range(len(present))), ",".join(l for _, l in present)))
s.append("\\addplot[fill=chot!25, draw=chot] coordinates {%s}; \\addlegendentry{bare policy}" % " ".join("(%d,%.1f)" % (i, reached(t, "_nf")) for i, (t, _) in enumerate(present)))
s.append("\\addplot[fill=chot, draw=chot] coordinates {%s}; \\addlegendentry{with the module}" % " ".join("(%d,%.1f)" % (i, reached(t, "_f")) for i, (t, _) in enumerate(present)))
s.append("\\panel{a}{}")
s.append("\\nextgroupplot[title={violations of the bare policy}, xtick={%s}, xticklabels={%s}, ylabel={samples in violation [\\%%]}, ymax=3.5]" % (",".join(str(i) for i in range(len(present))), ",".join(l for _, l in present)))
s.append("\\addplot[fill=black!20, draw=black!70] coordinates {%s};" % " ".join("(%d,%.2f)" % (i, viol(t)) for i, (t, _) in enumerate(present)))
s.append("\\panel{b}{}")
present2 = [(t, l) for t, l in groups[1][1] if t + "_nf" in V]
s.append("\\nextgroupplot[title={%s}, xtick={%s}, xticklabels={%s}, ylabel={reached within 20 s [\\%%]}, ymax=105]" % (groups[1][0], ",".join(str(i) for i in range(len(present2))), ",".join(l for _, l in present2)))
s.append("\\addplot[fill=chot!25, draw=chot] coordinates {%s};" % " ".join("(%d,%.1f)" % (i, reached(t, "_nf")) for i, (t, _) in enumerate(present2)))
s.append("\\addplot[fill=chot, draw=chot] coordinates {%s};" % " ".join("(%d,%.1f)" % (i, reached(t, "_f")) for i, (t, _) in enumerate(present2)))
s.append("\\panel{c}{}")
s += ["\\end{groupplot}", "\\node[anchor=south, inner sep=1pt] at ($(group c1r1.north west)!0.5!(group c3r1.north east)+(0,9mm)$) {\\pgfplotslegendfromname{ablleg}};", "\\end{tikzpicture}", "\\end{document}"]
open(os.path.join(FS, "fig12_ablation.tex"), "w").write("\n".join(s)); print("written fig12_ablation.tex with", {k: len(v) for k, v in V.items()})
