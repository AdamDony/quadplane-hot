"""Controllers for the evaluation: HOT (network + backup safety filter), threshold rule, fixed lift-sharing schedule,
gain-scheduled PID. All expose step(x_meas, gd_meas, rho_t, vd_meas) -> u (3,) and reset(x0, u0)."""
import numpy as np, torch
from . import plant as PL, bsf, net as N
class Base:
    def reset(self, x0, u0, rho_t): self.u_prev = u0.copy(); self.rho_t = rho_t
    def clip(self, u):
        u = np.clip(u, self.u_prev - PL.DU, self.u_prev + PL.DU); return np.clip(u, PL.U_MIN, PL.U_MAX)
class Threshold(Base):
    """autopilot rule: front transition - pusher ramps at its rate limit, the rotors hold the flight path (PI on the
    flight-path rate) until the airspeed threshold, then are cut at their rate limit and the pitch setpoint takes the
    cruise value; back transition - pusher cut, rotors re-engaged at the threshold and holding the flight path,
    pitch setpoint level"""
    def __init__(self, V_trans=12.0): self.V_trans = V_trans; self.cut = False; self.ei = 0.0
    def reset(self, x0, u0, rho_t): super().reset(x0, u0, rho_t); self.cut = False; self.ei = 0.0
    def rotors_hold(self, gam, gd):
        e = -1.5 * gam - gd; self.ei += e * PL.DT
        return self.u_prev[2] + np.clip(40.0 * e + 20.0 * self.ei, -PL.DU[2], PL.DU[2])
    def step(self, x, gd, rho_t, vd=0.0):
        V, gam, th, Tc, Tr = x
        if rho_t > V:                                  # front
            if V >= self.V_trans: self.cut = True
            Tcc = self.u_prev[1] + PL.DU[1] if not self.cut else self.u_prev[1] + 1.5 * (rho_t - V) * PL.DT * 20
            Trc = self.u_prev[2] - PL.DU[2] if self.cut else self.rotors_hold(gam, gd)
            thsp = -0.09 if self.cut else 0.0
        else:                                          # back
            engaged = V <= self.V_trans
            Tcc = self.u_prev[1] - PL.DU[1] if not engaged else self.u_prev[1] + 1.5 * (rho_t - V) * PL.DT * 20
            Trc = self.rotors_hold(gam, gd) if engaged else self.u_prev[2]
            thsp = 0.0 if engaged else -0.09 * np.clip((V - 6) / 9, 0, 1)
        u = self.clip(np.array([thsp, Tcc, Trc])); self.u_prev = u; return u
class Schedule(Base):
    """fixed lift-sharing schedule: rotor thrust follows the nominal-trim schedule of the current airspeed, pusher PI on airspeed, pitch at the schedule value"""
    def __init__(self, k_v=3.0, k_i=0.8): self.k_v, self.k_i = k_v, k_i; self.ei = 0.0
    def reset(self, x0, u0, rho_t): super().reset(x0, u0, rho_t); self.ei = 0.0
    def step(self, x, gd, rho_t, vd=0.0):
        V, gam, th, Tc, Tr = x; xs, us = PL.trim(float(np.clip(V, 5.0, 16.0)), {k: PL.NOM[k] for k in PL.KEYS})
        e = rho_t - V; self.ei = float(np.clip(self.ei + e * PL.DT, -3.0, 3.0))           # anti-windup
        Tcc = us[1] + self.k_v * e + self.k_i * self.ei; Trc = us[2] + 30.0 * (-1.5 * gam - gd); thsp = us[0]
        u = self.clip(np.array([thsp, Tcc, Trc])); self.u_prev = u; return u
class GSPID(Base):
    """gain-scheduled PID: pusher PI on airspeed, rotors PI on flight-path rate with gains scheduled on airspeed, pitch scheduled on airspeed with a gamma term"""
    def __init__(self): self.ei = 0.0; self.eg = 0.0
    def reset(self, x0, u0, rho_t): super().reset(x0, u0, rho_t); self.ei = 0.0; self.eg = 0.0
    def step(self, x, gd, rho_t, vd=0.0):
        V, gam, th, Tc, Tr = x; e = rho_t - V; self.ei = float(np.clip(self.ei + e * PL.DT, -3.0, 3.0)); s = np.clip((V - 6.0) / 9.0, 0, 1)
        kp_v, ki_v = 4.0, 1.0; Tcc = self.u_prev[1] * 0.0 + PL.trim(float(np.clip(V, 5, 16)), {k: PL.NOM[k] for k in PL.KEYS})[1][1] + kp_v * e + ki_v * self.ei
        gd_ref = -1.5 * gam; eg = gd_ref - gd; self.eg = float(np.clip(self.eg + eg * PL.DT, -0.5, 0.5))
        Trc = self.u_prev[2] + (40.0 * (1 - 0.6 * s)) * eg * PL.DT * 20 + 5.0 * self.eg
        thsp = -0.09 * s + 0.5 * gam
        u = self.clip(np.array([thsp, Tcc, Trc])); self.u_prev = u; return u
class HOTController(Base):
    """the trained network with the sliding window and the backup safety filter"""
    def __init__(self, net, use_filter=True, reduced=True, n_samples=64):
        self.net = net.eval(); self.L = net.L; self.use_filter = use_filter; self.F = bsf.Filter(n_samples=n_samples, reduced=reduced) if use_filter else None
    def reset(self, x0, u0, rho_t):
        super().reset(x0, u0, rho_t); self.xw = np.tile(x0, (self.L, 1)); self.uw = np.tile(u0, (self.L, 1)); self.x_last = x0.copy(); self.u_last = u0.copy(); self.first = True
        if self.F is not None: self.F.reset()
    def step(self, x, gd, rho_t, vd=0.0):
        if self.F is not None and not self.first: self.F.observe(self.x_last, self.u_last, x)
        self.first = False; self.xw = np.vstack([self.xw[1:], x[None]])
        with torch.no_grad():
            u_nn, theta_hat, _ = self.net(torch.tensor(self.xw[None], dtype=torch.float32), torch.tensor(self.uw[None], dtype=torch.float32), torch.tensor([rho_t], dtype=torch.float32))
        u_nn = u_nn[0].numpy().astype(float); self.theta_hat = theta_hat[0].numpy()
        u = self.clip(u_nn)
        if self.F is not None: u, act = self.F(x, self.u_prev, u, gd, vd)
        self.uw = np.vstack([self.uw[1:], u[None]]); self.u_prev = u; self.x_last = x.copy(); self.u_last = u.copy(); return u
class RLController(Base):
    """Stable-Baselines3 policy with the environment's observation window"""
    def __init__(self, model, L=8): self.model = model; self.L = L
    def reset(self, x0, u0, rho_t): super().reset(x0, u0, rho_t); self.xw = np.tile(x0, (self.L, 1)); self.uw = np.tile(u0, (self.L, 1))
    def step(self, x, gd, rho_t, vd=0.0):
        from .rl_env import XS, US
        self.xw = np.vstack([self.xw[1:], x[None]]); obs = np.clip(np.concatenate([(self.xw / XS).ravel(), (self.uw / US).ravel(), [rho_t / 10.0]]).astype(np.float32), -10, 10)
        a, _ = self.model.predict(obs, deterministic=True); u = self.clip(self.u_prev + PL.DU * np.clip(a, -1, 1)); self.uw = np.vstack([self.uw[1:], u[None]]); self.u_prev = u; return u
