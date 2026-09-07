"""PPO / SAC baselines with Stable-Baselines3 on the hand-off environment."""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hot.rl_env import HandoffEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
torch.set_num_threads(2)
ap = argparse.ArgumentParser(); ap.add_argument("--algo", default="ppo"); ap.add_argument("--steps", type=int, default=600000); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "..", "runs"); os.makedirs(OUT, exist_ok=True)
env = DummyVecEnv([lambda i=i: HandoffEnv(seed=a.seed * 100 + i) for i in range(4)]) if a.algo == "ppo" else HandoffEnv(seed=a.seed)
if a.algo == "ppo": model = PPO("MlpPolicy", env, n_steps=1024, batch_size=256, learning_rate=3e-4, gamma=0.995, gae_lambda=0.97, ent_coef=0.0, policy_kwargs=dict(net_arch=[128, 128]), seed=a.seed, verbose=0)
else: model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=300000, batch_size=256, gamma=0.995, train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[128, 128]), seed=a.seed, verbose=0)
from stable_baselines3.common.callbacks import BaseCallback
class Log(BaseCallback):
    def __init__(self): super().__init__(); self.n = 0
    def _on_step(self):
        self.n += 1
        if self.n % 20000 == 0:
            ep = [e for e in self.model.ep_info_buffer] if hasattr(self.model, "ep_info_buffer") and self.model.ep_info_buffer else []
            print("steps %d mean episode reward %.2f" % (self.num_timesteps, np.mean([e["r"] for e in ep]) if ep else float("nan")), flush=True)
        return True
model.learn(total_timesteps=a.steps, callback=Log()); model.save(os.path.join(OUT, f"{a.algo}_seed{a.seed}")); print("saved")
