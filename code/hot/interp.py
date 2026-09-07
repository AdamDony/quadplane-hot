"""Interpretability exports: attention of the context token over the window, R^2 of the context-token plant estimates against
time (the window slides, so the estimate of a parameter holds while the response that reveals it is inside the window), and
the estimates at the time of their best R^2 (raw and standardised)."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hot import plant as PL, controllers as C, net as N, evaluate as E, bsf
H = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(H, "..", "runs"); FD = os.path.join(H, "..", "..", "figdata"); RES = os.path.join(H, "..", "..", "results")
tag = sys.argv[1] if len(sys.argv) > 1 else "hot_v4"; sfx = "" if tag == "hot_v4" else "_" + tag
cfg = json.load(open(os.path.join(RUNS, f"{tag}_config.json"))); net = N.build("hot", L=cfg["L"], d=cfg["d"], n_blocks=cfg["blocks"], heads=cfg.get("heads", 4), use_phys=not cfg.get("no_phys", False)); net.load_state_dict(torch.load(os.path.join(RUNS, f"{tag}_best.pt"))); net.eval()
rng = np.random.default_rng(3); nP = 100; P = PL.sample_box(nP, rng); nom = {k: PL.NOM[k] for k in PL.KEYS}; KS = list(range(9, 300, 10)); KEYS = PL.KEYS
truth = np.array([[float(P[k][i]) for k in KEYS] for i in range(nP)]); lo = np.array([PL.BOX[k][0] for k in KEYS]); hi = np.array([PL.BOX[k][1] for k in KEYS])
out = {}; att = {}
for name, rho in (("front", (6.0, 15.0)), ("back", (15.0, 6.0))):
    est = np.zeros((len(KS), nP, len(KEYS))); ws_acc = []
    for i in range(nP):
        P1 = E.one_plant(P, i); ctrl = C.HOTController(net, use_filter=False); r = E.run_episode(ctrl, P1, rho[0], rho[1], n_steps=1, noise_scale=1.0, seed=i)   # reset with the nominal trim as in the campaigns
        x0, u0 = PL.trim(rho[0], nom); ctrl.reset(x0, u0, rho[1]); x = x0.copy(); rs = np.random.default_rng(i)
        for k in range(300):
            xm = x + rs.normal(0, 1, 5) * E.NOISE; gd = bsf.gamma_dot(x[None], P1)[0] + rs.normal(0, 0.02); u = ctrl.step(xm, gd, rho[1], 0.0); x = PL.step(x[None], u[None], P1)[0]
            if k in KS:
                with torch.no_grad(): _, th, ws = net(torch.tensor(ctrl.xw[None], dtype=torch.float32), torch.tensor(ctrl.uw[None], dtype=torch.float32), torch.tensor([rho[1]]), need_weights=True)
                est[KS.index(k), i] = th[0].numpy()
                if k == 59: ws_acc.append(ws[-1][0, 0, 1:].numpy())
    r2 = np.zeros((len(KS), len(KEYS)))
    for a in range(len(KS)):
        for j in range(len(KEYS)): r2[a, j] = float(np.corrcoef(truth[:, j], est[a, :, j])[0, 1] ** 2) if np.std(est[a, :, j]) > 1e-9 else 0.0
    out[name] = dict(r2=r2, est=est); att[name] = np.mean(ws_acc, 0)
with open(os.path.join(FD, f"ctx_r2_time{sfx}.dat"), "w") as fh:
    fh.write("t " + " ".join(f"{k}_{n}" for n in ("front", "back") for k in KEYS) + "\n")
    for a, k in enumerate(KS): fh.write("%.2f " % ((k + 1) * PL.DT) + " ".join("%.4f" % out[n]["r2"][a, j] for n in ("front", "back") for j in range(len(KEYS))) + "\n")
best = {}
with open(os.path.join(FD, f"ctx_estimates{sfx}.dat"), "w") as fh:
    fh.write("param truth est truth_n est_n\n")
    for j, k in enumerate(KEYS):
        cand = [(out[n]["r2"][a, j], n, a) for n in ("front", "back") for a in range(len(KS))]; r2b, nb, ab = max(cand); e = out[nb]["est"][ab, :, j]
        best[k] = dict(r2=round(r2b, 3), transition=nb, t=round((KS[ab] + 1) * PL.DT, 2), shrink=round(float(e.std() / (truth[:, j].std())), 3))
        for i in range(nP): fh.write("%s %.5f %.5f %.4f %.4f\n" % (k, truth[i, j], e[i], (truth[i, j] - 0.5 * (lo[j] + hi[j])) / (0.5 * (hi[j] - lo[j])), (e[i] - e.mean()) / max(e.std(), 1e-9)))
with open(os.path.join(FD, f"ctx_attention{sfx}.dat"), "w") as fh:
    fh.write("lag front back\n"); af, ab = att["front"], att["back"]
    for j in range(len(af)): fh.write("%d %.5f %.5f\n" % (j - len(af) + 1, af[j], ab[j]))
r2_end = {k: float(max(out["front"]["r2"][-1, j], out["back"]["r2"][-1, j])) for j, k in enumerate(KEYS)}
json.dump(dict(r2=r2_end, best=best, att_max=float(max(af.max(), ab.max())), att_min=float(min(af.min(), ab.min())), att_uniform=1.0 / len(af), baseline_err=0.25), open(os.path.join(RES, f"interp{sfx.replace('_hot', '')}.json"), "w"), indent=1)
print("best R2 per parameter:", best); print("attention range %.4f-%.4f (uniform %.4f)" % (min(af.min(), ab.min()), max(af.max(), ab.max()), 1.0 / len(af)))
