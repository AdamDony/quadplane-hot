#!/bin/bash
# nominal trajectories of the MLP and LSTM baselines for Fig. 6, then rebuild the figure and the manuscript once the main pipeline is done
PY=${PY:-python}; cd "$(dirname "$0")"; export OPENBLAS_NUM_THREADS=1
while [ ! -f runs/mlp_v2_last.pt ]; do sleep 60; done
$PY hot/campaign.py mlp nominal --save_traj > ../results/campaign_mlp_nominal.log 2>&1 &
$PY hot/campaign.py lstm nominal --save_traj > ../results/campaign_lstm_nominal.log 2>&1 &
wait
while [ ! -f ../results/final_done.txt ]; do sleep 60; done
$PY hot/export_figdata.py > ../results/export_fig06.log 2>&1
cd ../figsrc && pdflatex -interaction=nonstopmode -halt-on-error fig06_nominal.tex > fig06_nominal.log 2>&1 && cp fig06_nominal.pdf ../paper/figures/ || echo "fig06 failed"
cd ../paper && pdflatex -interaction=nonstopmode main.tex > main.log 2>&1 && pdflatex -interaction=nonstopmode main.tex > main.log 2>&1; echo done > ../results/fig06_done.txt
