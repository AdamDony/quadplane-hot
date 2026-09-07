"""Nominal-model MPC baseline: a tracking MPC with an artificial reference on a 4 s horizon, solved by sequential conic
programming (Clarabel), on the nominal identified model. Wraps the solver module of the author's earlier tooling."""
import os, sys, numpy as np
TCST = os.environ.get("QUADPLANE_MPC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "quadplane-mpc-handoff", "code"))   # solver of the MPC baseline: sibling clone of github.com/AdamDony/quadplane-mpc-handoff
sys.path.insert(0, TCST)
import mpc as _mpc
from . import plant as PL
class MPCController:
    def __init__(self, iters=3):
        self.iters = iters
    def reset(self, x0, u0, rho_t):
        self.u_prev = u0.copy(); self.rho_t = rho_t
        self.ctrl = _mpc.MPC(rho_t, self.iters, init_equilibrium=True, retighten=False)
        self.xh = x0.copy(); self.first = True
    def step(self, x, gd, rho_t, vd=0.0):
        try:
            u, _, _, ok = self.ctrl.solve(np.asarray(x, float))
        except Exception:
            u = self.u_prev.copy()
        u = np.clip(np.clip(u, self.u_prev - PL.DU, self.u_prev + PL.DU), PL.U_MIN, PL.U_MAX); self.u_prev = u; return u
