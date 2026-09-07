"""Generate figsrc/fig07_cross.tex (box plots per metric over the unseen plants) from figdata/box_cross.dat."""
import os, sys, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(H, "..", ".."); FD = os.path.join(ROOT, "figdata"); FS = os.path.join(ROOT, "figsrc")
ctrls = ["threshold", "schedule", "gspid", "mpc", "mlp", "lstm", "hot_nofilter", "ppo", "sac", "hot"]; labels = ["THR", "SCH", "PID", "MPC", "MLP", "LSTM", "H$-$L", "PPO", "SAC", "HOT"]
metrics = [("t_target", "time to target [s]", None), ("energy_Wh", "energy [Wh]", None), ("lf_min", "minimum lift fraction", 0.75), ("al_max_deg", "maximum $\\alpha$ [deg]", 5.0)]
def read(cell):
    d = {}
    for line in open(os.path.join(FD, f"box_{cell}.dat")).read().strip().split("\n")[1:]:
        p = line.split(); d[(p[1], p[2])] = [float(v) for v in p[3:8]] + [int(p[8]), float(p[9])]
    return d
def make(cell, out):
    d = read(cell); present = [c for c in ctrls if (c, "t_target") in d]; idx = {c: i + 1 for i, c in enumerate(present)}
    s = ["\\documentclass[tikz,border=2pt]{standalone}", "\\input{preamble}", "\\usepgfplotslibrary{statistics}", "\\definecolor{chot}{HTML}{C0392B}", "\\begin{document}", "\\begin{tikzpicture}",
         "\\begin{groupplot}[group style={group size=4 by 1, horizontal sep=9mm}, paperaxis, width=3.4cm, height=2.7cm, xmin=0.4, xmax=%d.6, xtick={1,...,%d}, xticklabels={%s}, x tick label style={rotate=60, anchor=east, font=\\tiny}, tick label style={font=\\tiny}, boxplot/draw direction=y, boxplot/box extend=0.55]" % (len(present), len(present), ",".join(labels[ctrls.index(c)] for c in present))]
    for j, (m, yl, bound) in enumerate(metrics):
        lim = {"t_target": ", ymin=0", "energy_Wh": ", ymin=0", "lf_min": ", ymin=0, ymax=1.25, restrict y to domain=-1:2", "al_max_deg": ", ymin=-12, ymax=22, restrict y to domain=-30:40"}[m]
        s.append("\\nextgroupplot[ylabel={%s}%s]" % (yl, lim))
        if bound is not None: s.append("\\draw[cmpc, dotted, line width=0.8pt] (axis cs:0.4,%g) -- (axis cs:%d.6,%g);" % (bound, len(present), bound))
        for c in present:
            lw, q1, md, q3, uw, n, reached = d[(c, m)]
            if not np.isfinite(md): continue
            col = "chot, fill=chot!20" if c == "hot" else "black, fill=black!10"
            s.append("\\addplot[boxplot prepared={draw position=%d, lower whisker=%.4f, lower quartile=%.4f, median=%.4f, upper quartile=%.4f, upper whisker=%.4f}, %s] coordinates {};" % (idx[c], lw, q1, md, q3, uw, col))
        s.append("\\panel{%s}{}" % "abcd"[j])
    s += ["\\end{groupplot}", "\\end{tikzpicture}", "\\end{document}"]
    open(os.path.join(FS, out), "w").write("\n".join(s)); print("written", out, "controllers", present)
if __name__ == "__main__":
    make("cross", "fig07_cross.tex")
    if os.path.exists(os.path.join(FD, "box_outbox.dat")): make("outbox", "fig08_outbox.tex")
