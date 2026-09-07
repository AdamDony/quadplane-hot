#!/bin/bash
# evaluation campaign for the listed controllers on the listed cells
PY=${PY:-python}; cd "$(dirname "$0")"; export OPENBLAS_NUM_THREADS=1
for c in "$@"; do $PY hot/campaign.py $c nominal cross outbox gust noise --save_traj > ../results/campaign_$c.log 2>&1; done
