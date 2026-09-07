# PX4 firmware in the loop

`sil_front.*` and `sil_back.*` are the logged front and back transitions of HOT with the PX4 v1.17 firmware in
software-in-the-loop (SIH standard VTOL airframe patched to the identified aircraft by `px4_sih_quadplane.patch`,
lockstep at quarter speed); `node_hot.log` is the console log of the node (`../hot/sil_node_hot.py`), `px4_sitl_hot.log`
the firmware console when the run script is used. `../hot/sil_analyze_hot.py` turns the logs into `results/sil_hot.json`
and the figure data of Fig. 10. Build PX4 as described in the sibling repository quadplane-mpc-handoff (`code/sil/README.md`
there), then `SPEED=0.25 bash ../run_sil_hot.sh`.
