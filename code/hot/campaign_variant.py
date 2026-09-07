"""Cross-plant cell (30 plants, both transitions) for a trained variant, evaluated WITHOUT the safety module (what the policy learned) and WITH it."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hot import plant as PL, controllers as C, evaluate as E, net as N
H = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(H, "..", "runs"); RES = os.path.join(H, "..", "..", "results")
tag = sys.argv[1]; cfg = json.load(open(os.path.join(RUNS, f"{tag}_config.json"))); arch = cfg.get("arch", "hot")
net = N.build(arch, L=cfg["L"], d=cfg["d"], n_blocks=cfg["blocks"], heads=cfg.get("heads", 4), use_phys=not cfg.get("no_phys", False)); net.load_state_dict(torch.load(os.path.join(RUNS, f"{tag}_best.pt" if os.path.exists(os.path.join(RUNS, f"{tag}_best.pt")) else f"{tag}_last.pt")))
rng = np.random.default_rng(0); P = PL.sample_box(100, rng); out = []
for use_filter in (False, True):
    for i in range(30):
        for rho in ((6.0, 15.0), (15.0, 6.0)):
            ctrl = C.HOTController(net, use_filter=use_filter); r = E.run_episode(ctrl, E.one_plant(P, i), rho[0], rho[1], n_steps=400)
            out.append(dict(ctrl=f"{tag}{'_f' if use_filter else '_nf'}", cell="ablation", rho0=rho[0], rhot=rho[1], plant_id=i, seed=0, **r["metrics"]))
            json.dump(out, open(os.path.join(RES, f"variant_{tag}.json"), "w"))
    print(tag, "filter" if use_filter else "bare", "reached %.2f t %.1f viol %.2f%% lf_min %.2f" % (np.mean([o["reached"] for o in out if o["ctrl"].endswith('_f' if use_filter else '_nf')]), np.nanmean([o["t_target"] for o in out if o["ctrl"].endswith('_f' if use_filter else '_nf')]), np.mean([o["viol_pct"] for o in out if o["ctrl"].endswith('_f' if use_filter else '_nf')]), np.mean([o["lf_min"] for o in out if o["ctrl"].endswith('_f' if use_filter else '_nf')])), flush=True)
