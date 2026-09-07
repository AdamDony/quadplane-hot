#!/bin/bash
# ablation and baseline trainings, run one after another once hot_v2 has finished
PY=${PY:-python}; cd "$(dirname "$0")"; export OPENBLAS_NUM_THREADS=1
while [ ! -f runs/hot_v4_last.pt ] || ! grep -q "campaign hot done" ../results/campaign_hot_nominal.log 2>/dev/null; do sleep 60; done
run() { tag=$1; shift; [ -f ../runs/${tag}_last.pt ] && return; $PY hot/train.py --tag $tag "$@" > ../runs/$tag.log 2>&1; }
run lstm_v2 --arch lstm --iters 800 --batch 32 --nmin 60 --nmax 300 --L 32 --lr 2e-4
run mlp_v2 --arch mlp --iters 800 --batch 32 --nmin 60 --nmax 300 --L 32 --lr 2e-4
run hot_nophys --arch hot --no_phys --iters 500 --batch 32 --nmin 60 --nmax 300 --L 32 --d 48 --lr 2e-4
run hot_noaux --arch hot --no_aux --iters 500 --batch 32 --nmin 60 --nmax 300 --L 32 --d 48 --lr 2e-4
run hot_L8 --arch hot --iters 500 --batch 32 --nmin 60 --nmax 300 --L 8 --d 48 --lr 2e-4
run hot_L16 --arch hot --iters 500 --batch 32 --nmin 60 --nmax 300 --L 16 --d 48 --lr 2e-4
run hot_L64 --arch hot --iters 500 --batch 32 --nmin 60 --nmax 300 --L 64 --d 48 --lr 2e-4
run hot_d32 --arch hot --iters 500 --batch 32 --nmin 60 --nmax 300 --L 32 --d 32 --lr 2e-4
run hot_d96 --arch hot --iters 500 --batch 32 --nmin 60 --nmax 300 --L 32 --d 96 --lr 2e-4
run hot_b1 --arch hot --iters 500 --batch 32 --nmin 60 --nmax 300 --L 32 --d 48 --blocks 1 --lr 2e-4
run hot_b4 --arch hot --iters 500 --batch 32 --nmin 60 --nmax 300 --L 32 --d 48 --blocks 4 --lr 2e-4
echo "queue done" > ../runs/queue_done.txt
