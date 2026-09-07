#!/bin/bash
# PX4 software-in-the-loop with HOT: launch the SITL (lockstep at half speed), run the offboard node, analyse
PY=${PY:-python}; C=$(cd "$(dirname "$0")" && pwd); export OPENBLAS_NUM_THREADS=1
pkill -f "sil_node" ; pkill -f "bin/px4 -d"; sleep 3
cd ${PX4_DIR:-$HOME/PX4-Autopilot}/build/px4_sitl_default/rootfs && rm -rf log dataman 2>/dev/null
(PX4_SIM_SPEED_FACTOR=${SPEED:-0.5} PX4_SIM_MODEL=sihsim_standard_vtol PX4_SIMULATOR=sihsim ../bin/px4 -d > $C/sil/px4_sitl_hot.log 2>&1 &)
for i in $(seq 1 60); do grep -q "Ready for takeoff" $C/sil/px4_sitl_hot.log 2>/dev/null && break; sleep 2; done
cd $C && mkdir -p sil && MAVLINK20=1 MAVLINK_DIALECT=common $PY -u hot/sil_node_hot.py both --log $C/sil > sil/node_hot.log 2>&1
echo "node exit $?" >> sil/node_hot.log; pkill -f "bin/px4 -d"
$PY hot/sil_analyze_hot.py >> sil/node_hot.log 2>&1
