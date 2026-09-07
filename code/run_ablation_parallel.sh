#!/bin/bash
# reduced ablation: five HOT variants trained in parallel after the main campaign, then the variant campaigns in parallel, figure and numbers
PY=${PY:-python}; cd "$(dirname "$0")"; export OPENBLAS_NUM_THREADS=1
while [ ! -f ../results/final_done.txt ]; do sleep 60; done
pkill -f run_queue.sh; pkill -f run_ablation_campaign.sh; sleep 2
# let a running mlp_v2 finish, kill any other queue training
for t in hot_nophys hot_noaux hot_L8 hot_L16 hot_L64 hot_d32 hot_d96 hot_b1 hot_b4; do pkill -f "train.py --tag $t" ; done; sleep 2
while pgrep -f "train.py --tag mlp_v2" > /dev/null; do sleep 60; done
run() { tag=$1; shift; [ -f ../runs/${tag}_last.pt ] && return; $PY hot/train.py --tag $tag "$@" > ../runs/$tag.log 2>&1; }
run hot_nophys --arch hot --no_phys --iters 500 --batch 32 --nmin 60 --nmax 300 --L 32 --d 48 --lr 2e-4 &
run hot_noaux --arch hot --no_aux --iters 500 --batch 32 --nmin 60 --nmax 300 --L 32 --d 48 --lr 2e-4 &
run hot_L8 --arch hot --iters 500 --batch 32 --nmin 60 --nmax 300 --L 8 --d 48 --lr 2e-4 &
run hot_d32 --arch hot --iters 500 --batch 32 --nmin 60 --nmax 300 --L 32 --d 32 --lr 2e-4 &
run hot_b1 --arch hot --iters 500 --batch 32 --nmin 60 --nmax 300 --L 32 --d 48 --blocks 1 --lr 2e-4 &
# variant campaigns of the already trained networks meanwhile
for tag in hot_v4 mlp_v2 lstm_v2; do $PY hot/campaign_variant.py $tag > ../results/campaign_variant_$tag.log 2>&1 & done
wait
for tag in hot_nophys hot_noaux hot_L8 hot_d32 hot_b1; do $PY hot/campaign_variant.py $tag > ../results/campaign_variant_$tag.log 2>&1 & done
$PY hot/interp.py hot_noaux > ../results/interp_noaux.log 2>&1 &
wait
$PY hot/make_ablfig.py > ../results/ablfig.log 2>&1; $PY hot/gen_numbers.py >> ../results/stats.log 2>&1
cd ../figsrc && pdflatex -interaction=nonstopmode -halt-on-error fig12_ablation.tex > fig12_ablation.log 2>&1 && cp fig12_ablation.pdf ../paper/figures/
cd ../paper && pdflatex -interaction=nonstopmode main.tex > main.log 2>&1 && pdflatex -interaction=nonstopmode main.tex > main.log 2>&1
echo done > ../results/ablation_done.txt
