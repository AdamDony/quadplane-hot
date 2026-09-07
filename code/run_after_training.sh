#!/bin/bash
# once hot_v2 has finished: HOT campaigns (parallel), HOT without the module, interpretability, timing, statistics, figure data
PY=${PY:-python}; cd "$(dirname "$0")"; export OPENBLAS_NUM_THREADS=1
while [ ! -f runs/hot_v4_last.pt ]; do sleep 60; done
bash run_hot_campaign.sh hot
$PY hot/campaign.py hot_nofilter nominal cross --save_traj > ../results/campaign_hot_nofilter.log 2>&1 &
( [ -f runs/ppo_seed1.zip ] && $PY hot/campaign.py ppo nominal cross outbox gust noise replay --save_traj > ../results/campaign_ppo.log 2>&1; [ -f runs/sac_seed0.zip ] && $PY hot/campaign.py sac nominal cross outbox gust noise replay --save_traj > ../results/campaign_sac.log 2>&1 ) &
$PY hot/interp.py hot_v4 > ../results/interp.log 2>&1
$PY hot/timing.py hot_v4 > ../results/timing.log 2>&1
wait
$PY hot/stats.py > ../results/stats.log 2>&1; $PY hot/export_figdata.py >> ../results/stats.log 2>&1; $PY hot/gen_tables.py >> ../results/stats.log 2>&1; $PY hot/gen_numbers.py >> ../results/stats.log 2>&1
echo "after-training pipeline done" > ../results/after_training_done.txt
