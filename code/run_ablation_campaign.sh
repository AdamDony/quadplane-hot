#!/bin/bash
# after the ablation trainings: cross-plant cell on 30 plants for every variant (network alone, no module: what the policy learned) and with the module for the module-removal question
PY=${PY:-python}; cd "$(dirname "$0")"; export OPENBLAS_NUM_THREADS=1
while [ ! -f ../runs/queue_done.txt ]; do sleep 120; done
for tag in hot_v4 mlp_v2 lstm_v2 hot_nophys hot_noaux hot_L8 hot_L16 hot_L64 hot_d32 hot_d96 hot_b1 hot_b4; do
  $PY hot/campaign_variant.py $tag > ../results/campaign_variant_$tag.log 2>&1
done
$PY hot/interp.py hot_noaux > ../results/interp_noaux.log 2>&1
echo done > ../results/ablation_done.txt
