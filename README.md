# Safe hand-off control of a quadplane VTOL with a history-conditioned transformer (HOT)

Author: Md Nur-A-Adam Dony ([AdamDony](https://github.com/AdamDony)).

Controller, lift-safety module, physics-informed training, baselines, evaluation campaigns with every result, PX4
software-in-the-loop set-up with its logs, figure sources and article sources of

> M. N.-A.-A. Dony and M. A. Islam, "Safe hand-off control of a quadplane VTOL across an uncertain plant family with a
> history-conditioned transformer," submitted to *Neurocomputing*, 2026.

Every number, table and figure of the article is produced by the scripts in this repository from the files in `results/`,
`figdata/` and `code/runs/`; the commands below regenerate them. The flight-test telemetry and the CFD data of the
aircraft are in the dataset repository [quadplane-px4-flight-data](https://github.com/AdamDony/quadplane-px4-flight-data);
the identified plant parameters derived from them are in `code/ident.json`. The nominal-model MPC baseline and the MAVLink
helpers of the software-in-the-loop node come from the sibling repository
[quadplane-mpc-handoff](https://github.com/AdamDony/quadplane-mpc-handoff): clone it next to this one or point
`QUADPLANE_MPC` at its `code/` folder.

## Layout

| path | content |
|---|---|
| `code/hot/plant.py`, `plant_torch.py` | five-state longitudinal model (NumPy evaluation plant with post-stall fold; differentiable PyTorch model), parameter box, trims |
| `code/hot/net.py` | HOT (window encoder SWE, plant-inference attention PIA, bounded head BPH), MLP and LSTM policies, Lipschitz bound |
| `code/hot/bsf.py` | lift-safety module LSM: backup law, settled neighbourhood, constraints, set-membership contraction, certification and filter |
| `code/hot/train.py` | physics-informed training by truncated backpropagation through the model (domain randomisation, gusts, noise, curriculum, validation-based checkpoint) |
| `code/hot/controllers.py`, `mpc_baseline.py`, `rl_env.py`, `rl_train.py` | threshold rule, lift-sharing schedule, gain-scheduled PID, nominal-model MPC, PPO/SAC agents |
| `code/hot/evaluate.py`, `campaign.py`, `campaign_variant.py` | closed-loop episodes and the evaluation cells (nominal, 100 unseen plants, out-of-box, gusts, noise, recorded flight states, ablation variants) |
| `code/hot/stats.py`, `gen_tables.py`, `gen_numbers.py`, `export_figdata.py`, `make_boxfig.py`, `make_ablfig.py`, `interp.py`, `timing.py` | statistics (Wilcoxon, Friedman), tables, number macros, figure data, interpretability and timing |
| `code/hot/sil_node_hot.py`, `sil_analyze_hot.py`, `code/sil/` | PX4 firmware-in-the-loop node, its logs and the SIH airframe patch |
| `code/run_*.sh` | the pipelines that produced the article: training, campaigns, ablation, SIL, final assembly |
| `code/runs/` | trained weights, configurations and training histories (`hot_v4` is the article's HOT; `mlp_v2`, `lstm_v2` and the `hot_*` variants are the baselines and ablations; PPO/SAC agents) |
| `results/` | every campaign result (JSON rows per episode), trajectories (`traj_*.npz`), statistics, timing, interpretability, SIL summary, logs |
| `figdata/`, `figsrc/` | pgfplots data and TikZ sources of every figure |
| `paper/`, `submission/` | article sources with the generated `numbers.tex` and `tables/`, the flat submission files and the compiled PDF |
| `design/` | design dossier of the architecture and the module, numerical verification of the settled neighbourhood |

## Reproducing the article

Python 3.10 with `numpy`, `scipy`, `torch`; `stable-baselines3` and `gymnasium` for the PPO/SAC baselines; `pymavlink`
for the software-in-the-loop node; `pdflatex` with `pgfplots` for the figures.

```bash
cd code
python hot/train.py --tag hot_v4 --iters 800 --batch 32 --nmin 60 --nmax 300 --L 32 --d 48 --lr 2e-4   # HOT (Fig. 5)
bash run_hot_campaign.sh hot; python hot/campaign.py hot_nofilter nominal cross --save_traj             # Tables 5-7, Figs. 6-9
python hot/campaign.py threshold nominal cross outbox gust noise replay --save_traj                       # baselines (likewise schedule, gspid, mpc, ppo, sac)
python hot/interp.py hot_v4; python hot/timing.py hot_v4                                                # Fig. 12, Table 8
python hot/stats.py; python hot/export_figdata.py; python hot/gen_tables.py; python hot/make_boxfig.py; python hot/gen_numbers.py
bash run_ablation_parallel.sh                                                                           # Fig. 11 (variants)
SPEED=0.25 bash run_sil_hot.sh                                                                          # Fig. 10 (needs PX4 v1.17 built with code/sil/px4_sih_quadplane.patch)
```

The settled-neighbourhood verification of Assumption 2 and the closed-loop tests of the module are in
`design/settle_numbers.json` (produced by the `settle` mode of the test script described in `code/hot/bsf.py`).
Figures: `cd figsrc && pdflatex figNN_*.tex`. Article: `cd paper && pdflatex main.tex` (three passes).
