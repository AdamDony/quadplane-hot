"""Differentiable identified model in PyTorch: batched RK4 step over a batch of plants (for training by BPTT)."""
import torch
from . import plant as PL
G, RHO, M, S = PL.G, PL.RHO, PL.M, PL.S
def f(x, u, P, gust):
    """x (n,5), u (n,3), P dict of (n,) tensors, gust (n,) vertical gust [m/s] -> dx/dt (n,5)"""
    V, gam, th, Tc, Tr = x.unbind(1); thsp, Tcc, Trc = u.unbind(1)
    Vs = torch.clamp(V, min=1.0); al = th - gam; al_aero = al + torch.atan2(gust, Vs)     # the gust changes the aerodynamic angle of the wing only
    qb = 0.5 * RHO * Vs ** 2; CL = P["CL0"] + P["CLa"] * al_aero; L = qb * S * CL; D = qb * S * (P["CD0"] + P["k"] * CL ** 2)
    ca, sa = torch.cos(al), torch.sin(al)
    dV = (Tc * ca - D - Tr * sa - M * G * torch.sin(gam)) / M
    dg = (L + Tc * sa + Tr * ca - M * G * torch.cos(gam)) / (M * Vs)
    dth = P["kappa"] * (P["g"] * thsp - th)
    dTc = P["lam_c"] * (P["beta_c"] * Tcc - Tc); dTr = P["lam_r"] * (P["beta_r"] * Trc - Tr)
    return torch.stack([dV, dg, dth, dTc, dTr], 1)
def step(x, u, P, gust=None, dt=PL.DT, sub=2):
    if gust is None: gust = torch.zeros_like(x[:, 0])
    h = dt / sub
    for _ in range(sub):
        k1 = f(x, u, P, gust); k2 = f(x + h / 2 * k1, u, P, gust); k3 = f(x + h / 2 * k2, u, P, gust); k4 = f(x + h * k3, u, P, gust)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
def lift_fraction(x, P):
    V, gam, th, Tc, Tr = x.unbind(1); al = th - gam; Vs = torch.clamp(V, min=1.0)
    L = 0.5 * RHO * Vs ** 2 * S * (P["CL0"] + P["CLa"] * al)
    return (L + Tc * torch.sin(al) + Tr * torch.cos(al)) / (M * G * torch.cos(gam))
def gamma_dot(x, P):
    V, gam, th, Tc, Tr = x.unbind(1); al = th - gam; Vs = torch.clamp(V, min=1.0)
    L = 0.5 * RHO * Vs ** 2 * S * (P["CL0"] + P["CLa"] * al)
    return (L + Tc * torch.sin(al) + Tr * torch.cos(al) - M * G * torch.cos(gam)) / (M * Vs)
def to_torch(P, dtype=torch.float32):
    return {k: torch.as_tensor(v, dtype=dtype) for k, v in P.items()}
