#!/bin/bash
# final pipeline after the module fix: HOT campaigns -> timing -> statistics, figure data, tables, numbers -> figures -> manuscript
PY=${PY:-python}; cd "$(dirname "$0")"; export OPENBLAS_NUM_THREADS=1
$PY - <<'PYEOF'
import glob, os
for f in glob.glob('../results/hot_cross_*.json') + glob.glob('../results/hot_gust_noise.json') + glob.glob('../results/hot_nominal_replay.json') + glob.glob('../results/hot_outbox.json') + glob.glob('../results/traj_hot_1*.npz') + glob.glob('../results/traj_hot_6*.npz') + glob.glob('../results/traj_hot_replay_*.npz'):
    os.remove(f); print('removed', f)
PYEOF
bash run_hot_campaign.sh hot
$PY hot/timing.py hot_v4 > ../results/timing.log 2>&1
$PY hot/stats.py > ../results/stats.log 2>&1; $PY hot/export_figdata.py >> ../results/stats.log 2>&1; $PY hot/gen_tables.py >> ../results/stats.log 2>&1; $PY hot/make_boxfig.py >> ../results/stats.log 2>&1; $PY hot/gen_numbers.py >> ../results/stats.log 2>&1
cd ../figsrc; for f in fig05_training fig06_nominal fig07_cross fig08_outbox fig09_sweeps fig10_replay fig13_interp; do [ -f $f.tex ] && pdflatex -interaction=nonstopmode -halt-on-error $f.tex > $f.log 2>&1 && cp $f.pdf ../paper/figures/ || echo "FIGURE FAILED $f"; done
cd ../paper; for i in 1 2 3; do pdflatex -interaction=nonstopmode main.tex > main.log 2>&1; done; bibtex main > /dev/null 2>&1; pdflatex -interaction=nonstopmode main.tex > main.log 2>&1; pdflatex -interaction=nonstopmode main.tex > main.log 2>&1
grep -c "^!" main.log; grep "Output written" main.log; echo "final pipeline done" > ../results/final_done.txt
