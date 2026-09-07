"""Evaluation campaign: cells (nominal, cross-plant, out-of-box, gust, noise, replay) for a named controller; results as JSON."""
import os, sys, json, time, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hot import plant as PL, controllers as C, evaluate as E, net as N
H = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(H, "..", "runs"); RES = os.path.join(H, "..", "results"); os.makedirs(RES, exist_ok=True)
def make(name, use_filter=True):
    if name == "threshold": return C.Threshold()
    if name == "schedule": return C.Schedule()
    if name == "gspid": return C.GSPID()
    if name in ("hot", "hot_nofilter", "mlp", "lstm"):
        arch = "hot" if name.startswith("hot") else name; cfg = json.load(open(os.path.join(RUNS, f"{arch}_v3_config.json" if arch == "hot" else f"{arch}_v2_config.json")))
        net = N.build(arch, L=cfg["L"], d=cfg["d"], n_blocks=cfg["blocks"], heads=cfg.get("heads", 4)); tag = "hot_v4" if arch == "hot" else f"{arch}_v2"
        sd = torch.load(os.path.join(RUNS, f"{tag}_best.pt" if os.path.exists(os.path.join(RUNS, f"{tag}_best.pt")) else f"{tag}_last.pt")); net.load_state_dict(sd)
        return C.HOTController(net, use_filter=(name != "hot_nofilter") and use_filter)
    if name == "mpc":
        from hot.mpc_baseline import MPCController; return MPCController(iters=3)
    if name in ("ppo", "sac"):
        from stable_baselines3 import PPO, SAC; from hot.rl_env import HandoffEnv
        model = (PPO if name == "ppo" else SAC).load(os.path.join(RUNS, "ppo_seed1" if name == "ppo" else "sac_seed0")); return C.RLController(model)
    raise ValueError(name)
def cells(which, n_plants=100, seed=0):
    rng = np.random.default_rng(seed); out = []
    if which == "nominal": out = [dict(cell="nominal", P=PL.nominal(1), rho=(6.0, 15.0), gust=0.0, noise=0.0), dict(cell="nominal", P=PL.nominal(1), rho=(15.0, 6.0), gust=0.0, noise=0.0)]
    elif which == "cross": 
        P = PL.sample_box(n_plants, rng); out = [dict(cell="cross", P=E.one_plant(P, i), rho=r, gust=0.0, noise=0.0, plant_id=i) for i in range(n_plants) for r in ((6.0, 15.0), (15.0, 6.0))]
    elif which == "outbox":
        P = PL.sample_box(n_plants // 2, rng, scale=1.5); out = [dict(cell="outbox", P=E.one_plant(P, i), rho=r, gust=0.0, noise=0.0, plant_id=i) for i in range(n_plants // 2) for r in ((6.0, 15.0), (15.0, 6.0))]
    elif which == "gust": out = [dict(cell=f"gust{g}", P=PL.nominal(1), rho=r, gust=g, noise=0.0, seed=s) for g in (0.25, 0.5, 1.0, 1.5) for s in range(10) for r in ((6.0, 15.0), (15.0, 6.0))]
    elif which == "noise": out = [dict(cell=f"noise{nz}", P=PL.nominal(1), rho=r, gust=0.0, noise=nz, seed=s) for nz in (0.5, 1.0, 2.0) for s in range(10) for r in ((6.0, 15.0), (15.0, 6.0))]
    elif which == "replay":
        rp = np.load(os.path.join(H, "replay_states.npz")); tt, X = rp["t"], rp["X"]
        for name, t_start, rho in (("front", 1159.0, (6.0, 15.0)), ("back", 1404.0, (15.0, 6.0))):
            x0 = X[int(np.argmin(np.abs(tt - t_start)))].copy(); x0[0] = max(x0[0], 5.0); u0 = np.clip(np.array([x0[2], x0[3], x0[4]]), PL.U_MIN, PL.U_MAX)
            out.append(dict(cell=f"replay_{name}", P=PL.nominal(1), rho=rho, gust=0.0, noise=0.0, x0=x0, u0=u0))
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ctrl"); ap.add_argument("cells", nargs="+"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--steps", type=int, default=400); ap.add_argument("--save_traj", action="store_true"); ap.add_argument("--plant_range", type=int, nargs=2, default=None); ap.add_argument("--suffix", default="")
    a = ap.parse_args(); results = []; t0 = time.time()
    for which in a.cells:
        for c in cells(which, a.n):
            if a.plant_range is not None and c.get("plant_id", -1) >= 0 and not (a.plant_range[0] <= c["plant_id"] < a.plant_range[1]): continue
            ctrl = make(a.ctrl)
            r = E.run_episode(ctrl, c["P"], c["rho"][0], c["rho"][1], n_steps=a.steps, gust_sigma=c["gust"], noise_scale=c["noise"], seed=c.get("seed", 0), x0=c.get("x0"), u0=c.get("u0"))
            rec = dict(ctrl=a.ctrl, cell=c["cell"], rho0=c["rho"][0], rhot=c["rho"][1], plant_id=c.get("plant_id", -1), seed=c.get("seed", 0), **r["metrics"])
            if a.save_traj and c["cell"] == "nominal": np.savez(os.path.join(RES, f"traj_{a.ctrl}_{int(c['rho'][0])}to{int(c['rho'][1])}.npz"), X=r["X"], U=r["U"], LF=r["LF"], GD=r["GD"])
            if c["cell"].startswith("replay"): np.savez(os.path.join(RES, f"traj_{a.ctrl}_{c['cell']}.npz"), X=r["X"], U=r["U"], LF=r["LF"], GD=r["GD"])
            results.append(rec); print("%s %s %2.0f->%2.0f plant %3d seed %d: t %5.1f energy %.2f lf_min %.2f al_max %.1f viol %.1f%% %s" % (a.ctrl, c["cell"], c["rho"][0], c["rho"][1], rec["plant_id"], rec["seed"], rec["t_target"], rec["energy_Wh"], rec["lf_min"], rec["al_max_deg"], rec["viol_pct"], ("interv %d unver %d" % (rec.get("interventions", 0), rec.get("unverified", 0))) if "interventions" in rec else ""), flush=True)
            json.dump(results, open(os.path.join(RES, f"{a.ctrl}_{'_'.join(a.cells)}{a.suffix}.json"), "w"))
    print("done %.0f s" % (time.time() - t0))
if __name__ == "__main__": main()
