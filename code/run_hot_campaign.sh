#!/bin/bash
# HOT campaigns split over processes (the safety module makes each episode 30-60 s)
PY=${PY:-python}; cd "$(dirname "$0")"; export OPENBLAS_NUM_THREADS=1
C=${1:-hot}
$PY hot/campaign.py $C nominal replay --save_traj > ../results/campaign_${C}_nominal.log 2>&1 &
$PY hot/campaign.py $C cross --plant_range 0 50 --suffix _a > ../results/campaign_${C}_cross_a.log 2>&1 &
$PY hot/campaign.py $C cross --plant_range 50 100 --suffix _b > ../results/campaign_${C}_cross_b.log 2>&1 &
$PY hot/campaign.py $C outbox > ../results/campaign_${C}_outbox.log 2>&1 &
$PY hot/campaign.py $C gust noise > ../results/campaign_${C}_gustnoise.log 2>&1 &
wait; echo "campaign $C done" >> ../results/campaign_${C}_nominal.log
