"""Update-time benchmark: network evaluation (microseconds), safety check (milliseconds), baselines' step times."""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hot import plant as PL, controllers as C, net as N, bsf
H = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(H, "..", "runs"); RES = os.path.join(H, "..", "..", "results"); os.makedirs(RES, exist_ok=True)
torch.set_num_threads(1)
tag = sys.argv[1] if len(sys.argv) > 1 else "hot_v4"
cfg = json.load(open(os.path.join(RUNS, f"{tag}_config.json"))) if os.path.exists(os.path.join(RUNS, f"{tag}_config.json")) else dict(L=32, d=48, blocks=2, heads=4)
net = N.build("hot", L=cfg["L"], d=cfg["d"], n_blocks=cfg["blocks"], heads=cfg.get("heads", 4)); sd = os.path.join(RUNS, f"{tag}_best.pt"); net.load_state_dict(torch.load(sd)); net.eval()
nom = {k: PL.NOM[k] for k in PL.KEYS}; x0, u0 = PL.trim(6.0, nom)
xw = torch.tensor(np.tile(x0, (1, cfg["L"], 1)), dtype=torch.float32); uw = torch.tensor(np.tile(u0, (1, cfg["L"], 1)), dtype=torch.float32); rt = torch.tensor([15.0])
with torch.no_grad():
    for _ in range(50): net(xw, uw, rt)
    t = time.perf_counter(); n = 500
    for _ in range(n): net(xw, uw, rt)
    eval_us = 1e6 * (time.perf_counter() - t) / n
F = bsf.Filter(n_samples=32, reduced=True); ts = []
for V0 in (6.0, 8.0, 10.0, 12.0, 15.0):
    x, u = PL.trim(V0, nom)
    for _ in range(3): t = time.perf_counter(); F.check(x, u, u); ts.append(1e3 * (time.perf_counter() - t))
out = dict(hot_eval_us=eval_us, check_ms_mean=float(np.mean(ts)), check_ms_max=float(np.max(ts)), params=sum(p.numel() for p in net.parameters()), threads=1, machine="Apple M2, Python")
for name, ctrl in (("threshold", C.Threshold()), ("schedule", C.Schedule()), ("gspid", C.GSPID())):
    ctrl.reset(x0, u0, 15.0); t = time.perf_counter()
    for _ in range(200): ctrl.step(x0, 0.0, 15.0)
    out[f"{name}_us"] = 1e6 * (time.perf_counter() - t) / 200
json.dump(out, open(os.path.join(RES, "timing.json"), "w"), indent=1); print(out)
