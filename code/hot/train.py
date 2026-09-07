"""Physics-informed training of HOT by backpropagation through the differentiable identified model (Algorithm 1)."""
import os, sys, json, time, math, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hot import plant as PL, plant_torch as PT, net as N
torch.set_num_threads(4)
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "..", "runs")
KEYS = PL.KEYS; LO = torch.tensor([PL.BOX[k][0] for k in KEYS]); HI = torch.tensor([PL.BOX[k][1] for k in KEYS])
NOISE = torch.tensor([0.3, np.radians(0.5), np.radians(0.3), 0.6, 2.4], dtype=torch.float32)      # sensor noise std on (V, gam, th, Tc, Tr)
def sample_episodes(b, rng, scale=1.0):
    """plants, initial trims, targets, gust profiles for a batch"""
    P = PL.sample_box(b, rng, scale); Pt = PT.to_torch(P)
    kind = rng.uniform(size=b); rho0 = np.where(kind < 0.4, rng.uniform(5.5, 8.0, b), np.where(kind < 0.8, rng.uniform(13.0, 16.0, b), rng.uniform(6.0, 15.0, b)))
    rhot = np.where(kind < 0.4, 15.0, np.where(kind < 0.8, 6.0, rng.uniform(6.0, 15.0, b)))
    nom = {k: PL.NOM[k] for k in KEYS}; x0 = np.stack([PL.trim(v, nom)[0] for v in rho0]); u0 = np.stack([PL.trim(v, nom)[1] for v in rho0])
    return Pt, torch.tensor(x0, dtype=torch.float32), torch.tensor(u0, dtype=torch.float32), torch.tensor(rhot, dtype=torch.float32)
def gust_profile(b, n, rng, sigma):
    """vertical gust: first-order coloured noise plus a random constant, per episode (n, b)"""
    g = np.zeros((n, b)); c = rng.uniform(-sigma, sigma, b); a = math.exp(-PL.DT / 1.0); w = rng.normal(0, sigma * math.sqrt(1 - a * a), (n, b)); z = np.zeros(b)
    for j in range(n): z = a * z + w[j]; g[j] = c + z
    return torch.tensor(g, dtype=torch.float32)
def losses(x, u, u_prev, rhot, P, theta_hat, theta_true, eps=0.04):
    V, gam, th, Tc, Tr = x.unbind(1); al = th - gam
    lf = PT.lift_fraction(x, P); gd = PT.gamma_dot(x, P)
    time_l = torch.sigmoid(((V - rhot).abs() - 0.3) / 0.2) + (V - rhot).abs()            # indicator of not being at the target plus the distance in m/s (gradient everywhere)
    energy = (Tr + Tc * V) / 1500.0
    sc = np.radians(2.0)
    bar = (torch.nn.functional.softplus(-(lf - (1 - PL.KL_NOM)) / eps) + torch.nn.functional.softplus(-(PL.AL_MAX - al) / sc) +
           torch.nn.functional.softplus(-(PL.GAM_MAX - gam) / sc) + torch.nn.functional.softplus(-(gam - PL.GAM_MIN) / sc) +
           torch.nn.functional.softplus(-(gd + 0.2) / 0.05) + torch.nn.functional.softplus(-(PL.TH_MAX - th) / sc) + torch.nn.functional.softplus(-(th - PL.TH_MIN) / sc))
    aux = (((theta_hat - theta_true) / (HI - LO)) ** 2).mean(1)
    smooth = (((u - u_prev) / N.U_SCALE) ** 2).sum(1) + torch.nn.functional.softplus((gam.abs() - np.radians(4.0)) / np.radians(2.0))   # command smoothness and a level-flight preference
    return time_l, energy, bar, aux, smooth
def rollout(net, Pt, x0, u0, rhot, n_steps, gust, noise=True, trunc=150, weights=(1.0, 0.2, 2.0, 1.0, 0.3), rng=None):
    b = x0.shape[0]; L = net.L; theta_true = torch.stack([Pt[k] for k in KEYS], 1)
    xw = x0[:, None].expand(b, L, 5).clone(); uw = u0[:, None].expand(b, L, 3).clone(); x = x0.clone(); u_prev = u0.clone()
    tot = torch.zeros(5); loss_acc = 0.0; logs = dict(V=[], lf=[], al=[], gd=[], u=[])
    for j in range(n_steps):
        xw_in = xw + (torch.randn_like(xw) * NOISE if noise else 0.0)
        u, theta_hat, _ = net(xw_in, uw, rhot)
        x_new = PT.step(x, u, Pt, gust[j])
        alive = ((x_new[:, 0] > 3.0) & (x_new[:, 1].abs() < np.radians(30)) & ((x_new[:, 2] - x_new[:, 1]) < np.radians(15)) & ((x_new[:, 2] - x_new[:, 1]) > np.radians(-25))).float().unsqueeze(1)
        x = alive * x_new + (1 - alive) * x.detach()                    # diverged episodes are frozen (and keep paying the barrier)
        tl, en, br, ax, sm = losses(x, u, u_prev, rhot, Pt, theta_hat, theta_true)
        terms = torch.stack([tl.mean(), en.mean(), br.mean(), ax.mean(), sm.mean()]); loss_acc = loss_acc + (torch.tensor(weights) * terms).sum() * PL.DT
        tot += terms.detach() * PL.DT
        xw = torch.cat([xw[:, 1:], x[:, None]], 1); uw = torch.cat([uw[:, 1:], u[:, None]], 1); u_prev = u
        logs["V"].append(x[:, 0].detach()); logs["lf"].append(PT.lift_fraction(x, Pt).detach()); logs["al"].append((x[:, 2] - x[:, 1]).detach()); logs["gd"].append(PT.gamma_dot(x, Pt).detach()); logs["u"].append(u.detach())
        if (j + 1) % trunc == 0 and j + 1 < n_steps:     # truncated backpropagation through time
            loss_acc.backward(); loss_acc = 0.0
            x = x.detach(); xw = xw.detach(); uw = uw.detach(); u_prev = u_prev.detach()
    if torch.is_tensor(loss_acc): loss_acc.backward()
    return tot, {k: torch.stack(v) for k, v in logs.items()}
def metrics(logs, rhot):
    V = logs["V"]; reached = ((V - rhot).abs() <= 0.3)
    n = V.shape[0]; first = torch.where(reached.any(0), reached.float().argmax(0).float() * PL.DT, torch.full((V.shape[1],), float("nan")))
    return dict(t_target=float(np.nanmean(first.numpy())), reached=float(reached.any(0).float().mean()), lf_min=float(logs["lf"].min()), al_max_deg=float(np.degrees(logs["al"].max())), sink_min=float(logs["gd"].min()))
def make_val_set(n_plants=8, seed=123):
    rng = np.random.default_rng(seed); P = PL.sample_box(n_plants, rng); nom = PL.nominal(1)
    return [(nom, 6.0, 15.0), (nom, 15.0, 6.0)] + [({k: v[i:i + 1] for k, v in P.items()}, r0, r1) for i in range(n_plants) for r0, r1 in ((6.0, 15.0), (15.0, 6.0))]
def validate(net, val_set):
    from hot import controllers as C, evaluate as E
    net.eval(); ts, viols, lfs = [], [], []
    with torch.no_grad():
        for P1, r0, r1 in val_set:
            m = E.run_episode(C.HOTController(net, use_filter=False), P1, r0, r1, n_steps=400)["metrics"]
            ts.append(m["t_target"] if np.isfinite(m["t_target"]) else 20.0); viols.append(m["viol_pct"]); lfs.append(m["lf_min"])
    net.train(); return float(np.mean(ts) + np.mean(viols)), dict(t_mean=round(float(np.mean(ts)), 2), reached=round(float(np.mean([t < 20 for t in ts])), 2), viol=round(float(np.mean(viols)), 2), lf_min=round(float(np.min(lfs)), 2))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=1500); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--tag", default="hot")
    ap.add_argument("--L", type=int, default=32); ap.add_argument("--d", type=int, default=48); ap.add_argument("--blocks", type=int, default=2); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--nmin", type=int, default=60); ap.add_argument("--nmax", type=int, default=400); ap.add_argument("--gust", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_aux", action="store_true"); ap.add_argument("--no_phys", action="store_true"); ap.add_argument("--report", type=int, default=10); ap.add_argument("--arch", default="hot"); ap.add_argument("--heads", type=int, default=4)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    net = N.build(a.arch, L=a.L, d=a.d, n_blocks=a.blocks, heads=a.heads, use_phys=not a.no_phys); opt = torch.optim.Adam(net.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters, eta_min=a.lr / 10)
    weights = (1.0, 0.2, 2.0, 0.0 if (a.no_aux or a.arch != 'hot') else 1.0, 0.3); hist = []; t0 = time.time(); best = 1e9; val_set = make_val_set()
    for it in range(a.iters):
        n_steps = int(a.nmin + (a.nmax - a.nmin) * min(1.0, it / (0.6 * a.iters)))
        Pt, x0, u0, rhot = sample_episodes(a.batch, rng); gust = gust_profile(a.batch, n_steps, rng, a.gust)
        opt.zero_grad(); tot, logs = rollout(net, Pt, x0, u0, rhot, n_steps, gust, weights=weights)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        m = metrics(logs, rhot); rec = dict(it=it, n=n_steps, time=float(tot[0]), energy=float(tot[1]), barrier=float(tot[2]), aux=float(tot[3]), smooth=float(tot[4]), **m, wall=time.time() - t0); hist.append(rec)
        if it % a.report == 0 or it == a.iters - 1:
            print("it %4d N=%3d | time %.2f energy %.2f barrier %.3f aux %.3f smooth %.3f | t_target %.1f reached %.2f lf_min %.2f al_max %.1f sink_min %.2f | %.0fs" % (it, n_steps, rec["time"], rec["energy"], rec["barrier"], rec["aux"], rec["smooth"], m["t_target"], m["reached"], m["lf_min"], m["al_max_deg"], m["sink_min"], rec["wall"]), flush=True)
            json.dump(hist, open(os.path.join(OUT, f"{a.tag}_hist.json"), "w"))
        if (it % 50 == 0 and it >= a.iters // 4) or it == a.iters - 1:       # validation: 20 s episodes on the nominal plant and 8 fixed plants, both directions, bare policy
            score, vs = validate(net, val_set); rec["val"] = vs
            print("   validation it %d: score %.2f: %s" % (it, score, vs), flush=True)
            if score < best: best = score; torch.save(net.state_dict(), os.path.join(OUT, f"{a.tag}_best.pt"))
    torch.save(net.state_dict(), os.path.join(OUT, f"{a.tag}_last.pt")); json.dump(dict(vars(a), lipschitz=net.lipschitz_bound(), params=sum(p.numel() for p in net.parameters())), open(os.path.join(OUT, f"{a.tag}_config.json"), "w"), indent=1)
    print("done; lipschitz bound", net.lipschitz_bound())
if __name__ == "__main__": main()
