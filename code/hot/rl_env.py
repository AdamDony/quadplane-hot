"""Gymnasium environment of the hand-off for the reinforcement-learning baselines (PPO, SAC): same plant family,
window observation, bounded-rate actions and the same cost terms as the physics-informed training."""
import numpy as np, gymnasium as gym
from gymnasium import spaces
from . import plant as PL, bsf
NOISE = np.array([0.3, np.radians(0.5), np.radians(0.3), 0.6, 2.4]); XS = np.array([10.0, 0.2, 0.2, 30.0, 120.0]); US = np.array([0.5, 30.0, 120.0])
class HandoffEnv(gym.Env):
    def __init__(self, L=8, n_steps=400, gust=1.0, seed=0):
        super().__init__(); self.L = L; self.n_steps = n_steps; self.gust_sigma = gust; self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-10, 10, (L * 8 + 1,), np.float32); self.action_space = spaces.Box(-1, 1, (3,), np.float32); self.nom = {k: PL.NOM[k] for k in PL.KEYS}
    def _obs(self):
        w = np.concatenate([(self.xw / XS).ravel(), (self.uw / US).ravel(), [self.rho_t / 10.0]]).astype(np.float32); return np.clip(w, -10, 10)
    def reset(self, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.P = PL.sample_box(1, self.rng); kind = self.rng.uniform()
        rho0 = self.rng.uniform(5.5, 8.0) if kind < 0.4 else (self.rng.uniform(13.0, 16.0) if kind < 0.8 else self.rng.uniform(6.0, 15.0))
        self.rho_t = 15.0 if kind < 0.4 else (6.0 if kind < 0.8 else self.rng.uniform(6.0, 15.0))
        x0, u0 = PL.trim(rho0, self.nom); self.x = x0.copy(); self.u_prev = u0.copy(); self.xw = np.tile(x0, (self.L, 1)); self.uw = np.tile(u0, (self.L, 1)); self.k = 0
        from .evaluate import gust_series; self.gust = gust_series(self.n_steps, self.rng, self.gust_sigma); return self._obs(), {}
    def step(self, a):
        u = np.clip(self.u_prev + PL.DU * np.clip(a, -1, 1), PL.U_MIN, PL.U_MAX)
        xn = PL.step(self.x[None], u[None], self.P, gust=np.array([self.gust[self.k]]))[0]
        alive = xn[0] > 3.0 and abs(xn[1]) < np.radians(30) and (xn[2] - xn[1]) < np.radians(15) and (xn[2] - xn[1]) > np.radians(-25)
        if alive: self.x = xn
        V, gam, th, Tc, Tr = self.x; al = th - gam; lf = PL.lift_fraction(self.x[None], self.P)[0]; gd = bsf.gamma_dot(self.x[None], self.P)[0]
        sp = lambda z: np.log1p(np.exp(-np.clip(z, -30, 30)))
        sc = np.radians(2.0); bar = sp((lf - (1 - PL.KL_NOM)) / 0.04) + sp((PL.AL_MAX - al) / sc) + sp((PL.GAM_MAX - gam) / sc) + sp((gam - PL.GAM_MIN) / sc) + sp((gd + 0.2) / 0.05) + sp((PL.TH_MAX - th) / sc) + sp((th - PL.TH_MIN) / sc)
        time_l = 1 / (1 + np.exp(-((abs(V - self.rho_t) - 0.3) / 0.2))) + abs(V - self.rho_t); energy = (Tr + Tc * V) / 1500.0; smooth = (((u - self.u_prev) / US) ** 2).sum() + sp(-(abs(gam) - np.radians(4.0)) / np.radians(2.0))
        r = -(1.0 * time_l + 0.2 * energy + 2.0 * bar + 0.3 * smooth) * PL.DT
        self.u_prev = u; self.xw = np.vstack([self.xw[1:], (self.x + self.rng.normal(0, 1, 5) * NOISE)[None]]); self.uw = np.vstack([self.uw[1:], u[None]]); self.k += 1
        return self._obs(), float(r), False, self.k >= self.n_steps or not alive, dict(lf=lf, al=al)
