"""HOT: sliding-window encoder (SWE), plant-inference attention (PIA), bounded policy head (BPH)."""
import math, torch, torch.nn as nn
from . import plant as PL
G, RHO, M, S = PL.G, PL.RHO, PL.M, PL.S
KEYS = PL.KEYS
BOX_LO = torch.tensor([PL.BOX[k][0] for k in KEYS]); BOX_HI = torch.tensor([PL.BOX[k][1] for k in KEYS])
# normalisation of the raw window features (V, gam, th, Tc, Tr, thsp, Tcc, Trc)
X_SCALE = torch.tensor([10.0, 0.2, 0.2, 30.0, 120.0]); U_SCALE = torch.tensor([0.5, 30.0, 120.0])
def physics_features(x):
    """x (..., 5) -> (..., 4): dynamic pressure/100, wing lift fraction of the nominal polar at the current alpha, lift fraction estimate incl. rotors, alpha"""
    V, gam, th, Tc, Tr = x.unbind(-1); al = th - gam; Vs = torch.clamp(V, min=1.0)
    qb = 0.5 * RHO * Vs ** 2; L = qb * S * (PL.NOM["CL0"] + PL.NOM["CLa"] * al)
    return torch.stack([qb / 100.0, L / (M * G), (L + Tc * torch.sin(al) + Tr * torch.cos(al)) / (M * G), al / 0.1], -1)
class SWE(nn.Module):
    def __init__(self, d_model, L, use_phys=True):
        super().__init__(); self.use_phys = use_phys; self.lin = nn.Linear(5 + 3 + 4, d_model); self.pos = nn.Parameter(torch.zeros(1, L, d_model)); self.ctx = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pos, std=0.02); nn.init.normal_(self.ctx, std=0.02)
    def forward(self, xw, uw):
        """xw (b, L, 5), uw (b, L, 3) -> tokens (b, L+1, d) with the context token first"""
        pf = physics_features(xw); feats = torch.cat([xw / X_SCALE, uw / U_SCALE, pf if self.use_phys else torch.zeros_like(pf)], -1)
        e = self.lin(feats) + self.pos[:, :xw.shape[1]]
        return torch.cat([self.ctx.expand(e.shape[0], -1, -1), e], 1)
class Block(nn.Module):
    def __init__(self, d, heads, d_ff, drop=0.0):
        super().__init__(); self.ln1 = nn.LayerNorm(d); self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True); self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
    def forward(self, z, need_weights=False):
        h = self.ln1(z); a, w = self.att(h, h, h, need_weights=need_weights, average_attn_weights=True); z = z + a
        z = z + self.ff(self.ln2(z)); return z, w
class PIA(nn.Module):
    def __init__(self, d, heads, n_blocks, d_ff):
        super().__init__(); self.blocks = nn.ModuleList([Block(d, heads, d_ff) for _ in range(n_blocks)]); self.ln = nn.LayerNorm(d)
        self.aux = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, len(KEYS)))
    def forward(self, tokens, need_weights=False):
        z = tokens; ws = []
        for blk in self.blocks:
            z, w = blk(z, need_weights); ws.append(w)
        z = self.ln(z); z_ctx, z_L = z[:, 0], z[:, -1]
        theta_hat = BOX_LO + (BOX_HI - BOX_LO) * torch.sigmoid(self.aux(z_ctx))    # parameter estimate inside the box
        return z_ctx, z_L, theta_hat, ws
class BPH(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(2 * d + 1 + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.th_max = float(PL.U_MAX[0]); self.du = torch.tensor([float(PL.DU[0]), float(PL.DU[1]), float(PL.DU[2])])
        self.u_lo = torch.tensor(PL.U_MIN, dtype=torch.float32); self.u_hi = torch.tensor(PL.U_MAX, dtype=torch.float32)
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)      # start as a hold policy (u = u_prev)
    def forward(self, z_ctx, z_L, rho_t, u_prev):
        y = torch.tanh(self.mlp(torch.cat([z_ctx, z_L, (rho_t / 10.0).unsqueeze(-1), u_prev / U_SCALE], -1)))
        u = u_prev + self.du * y                       # every command rate-limited by construction (pitch setpoint included)
        return torch.maximum(torch.minimum(u, self.u_hi), self.u_lo)
class HOT(nn.Module):
    def __init__(self, L=40, d=64, heads=4, n_blocks=2, d_ff=128, use_phys=True):
        super().__init__(); self.L = L; self.swe = SWE(d, L, use_phys); self.pia = PIA(d, heads, n_blocks, d_ff); self.bph = BPH(d)
    def forward(self, xw, uw, rho_t, need_weights=False):
        """xw (b, L, 5) window of states (oldest first), uw (b, L, 3) window of commands (uw[:, -1] = u_{k-1}), rho_t (b,)"""
        tokens = self.swe(xw, uw); z_ctx, z_L, theta_hat, ws = self.pia(tokens, need_weights)
        u = self.bph(z_ctx, z_L, rho_t, uw[:, -1]); return u, theta_hat, ws
    def lipschitz_bound(self):
        """product of spectral norms along the residual stream (Lemma on the Lipschitz constant); attention treated by its softmax bound"""
        with torch.no_grad():
            lip = torch.linalg.matrix_norm(self.swe.lin.weight, 2).item()
            for blk in self.pia.blocks:
                Wqkv = blk.att.in_proj_weight; d = Wqkv.shape[1]; Wq, Wk, Wv = Wqkv[:d], Wqkv[d:2*d], Wqkv[2*d:]
                Wo = blk.att.out_proj.weight; r = 4.0
                att_l = torch.linalg.matrix_norm(Wo, 2).item() * torch.linalg.matrix_norm(Wv, 2).item() * (1 + 2 * r ** 2 * torch.linalg.matrix_norm(Wq, 2).item() * torch.linalg.matrix_norm(Wk, 2).item() / math.sqrt(d / blk.att.num_heads))
                ff_l = torch.linalg.matrix_norm(blk.ff[0].weight, 2).item() * torch.linalg.matrix_norm(blk.ff[2].weight, 2).item()
                lip *= (1 + att_l) * (1 + ff_l)
            for m in self.bph.mlp:
                if isinstance(m, nn.Linear): lip *= torch.linalg.matrix_norm(m.weight, 2).item()
            return lip * float(self.bph.du.max())
if __name__ == "__main__":
    net = HOT(); n = sum(p.numel() for p in net.parameters()); print("HOT parameters:", n)
    xw = torch.randn(2, 40, 5) * X_SCALE + torch.tensor([10, 0, 0, 10, 40.0]); uw = torch.zeros(2, 40, 3); u, th, _ = net(xw, uw, torch.tensor([15.0, 6.0])); print(u, th.shape, "lip", net.lipschitz_bound())

# ---------------- ablation architectures with the same interface ----------------
class MLPPolicy(nn.Module):
    """feed-forward policy on the current state, physics features, target and previous command (no history)"""
    def __init__(self, L=32, hidden=128):
        super().__init__(); self.L = L; self.mlp = nn.Sequential(nn.Linear(5 + 3 + 4 + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.du = torch.tensor([float(PL.DU[0]), float(PL.DU[1]), float(PL.DU[2])]); self.u_lo = torch.tensor(PL.U_MIN, dtype=torch.float32); self.u_hi = torch.tensor(PL.U_MAX, dtype=torch.float32)
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, xw, uw, rho_t, need_weights=False):
        x = xw[:, -1]; u_prev = uw[:, -1]; y = torch.tanh(self.mlp(torch.cat([x / X_SCALE, u_prev / U_SCALE, physics_features(x), (rho_t / 10.0).unsqueeze(-1)], -1)))
        u = torch.maximum(torch.minimum(u_prev + self.du * y, self.u_hi), self.u_lo); return u, BOX_LO + 0.5 * (BOX_HI - BOX_LO) * torch.ones(x.shape[0], len(KEYS)), None
    def lipschitz_bound(self): return float("nan")
class LSTMPolicy(nn.Module):
    """recurrent policy over the window (LSTM), bounded head"""
    def __init__(self, L=32, hidden=64):
        super().__init__(); self.L = L; self.lstm = nn.LSTM(5 + 3 + 4, hidden, batch_first=True); self.head = nn.Sequential(nn.Linear(hidden + 1 + 3, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.du = torch.tensor([float(PL.DU[0]), float(PL.DU[1]), float(PL.DU[2])]); self.u_lo = torch.tensor(PL.U_MIN, dtype=torch.float32); self.u_hi = torch.tensor(PL.U_MAX, dtype=torch.float32)
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self, xw, uw, rho_t, need_weights=False):
        feats = torch.cat([xw / X_SCALE, uw / U_SCALE, physics_features(xw)], -1); h, _ = self.lstm(feats); z = h[:, -1]; u_prev = uw[:, -1]
        y = torch.tanh(self.head(torch.cat([z, (rho_t / 10.0).unsqueeze(-1), u_prev / U_SCALE], -1)))
        u = torch.maximum(torch.minimum(u_prev + self.du * y, self.u_hi), self.u_lo); return u, BOX_LO + 0.5 * (BOX_HI - BOX_LO) * torch.ones(xw.shape[0], len(KEYS)), None
    def lipschitz_bound(self): return float("nan")
def build(arch, L=32, d=48, n_blocks=2, heads=4, use_phys=True):
    if arch == "hot": return HOT(L=L, d=d, n_blocks=n_blocks, heads=heads, use_phys=use_phys)
    if arch == "mlp": return MLPPolicy(L=L)
    if arch == "lstm": return LSTMPolicy(L=L)
    raise ValueError(arch)
