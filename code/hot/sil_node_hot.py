"""Algorithm 1 as a PX4 offboard node over MAVLink (software-in-the-loop with the PX4 firmware, SIH standard-VTOL plant).

The node closes the loop of Fig. 2 of the article around the real PX4 stack: it reads V, gamma, theta from the autopilot
(EKF2 + simulated sensors), propagates the thrust estimates (36) at the loop rate, re-plans every DP = 50 ms with the
same MPC class that produced the article's results, and sends the pitch setpoint to PX4's attitude loop
(SET_ATTITUDE_TARGET, rotor collective thrust) and the pusher command (MAV_CMD_DO_SET_ACTUATOR -> output function
"Peripheral via Actuator Set 1", assigned to the pusher output of the SIH standard VTOL).

usage: python sil_node.py front|back|both [--log DIR]   (PX4 SITL must be running: make px4_sitl sihsim_standard_vtol)
"""
import os, sys, time, json, math, argparse, struct
os.environ["MAVLINK20"] = "1"; os.environ["MAVLINK_DIALECT"] = "common"   # PX4 speaks MAVLink 2, common dialect
import numpy as np
from pymavlink import mavutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import torch, json
from hot import plant as PL, controllers as C, net as N, bsf
_sys.path.insert(0, _os.environ.get('QUADPLANE_MPC', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'quadplane-mpc-handoff', 'code')))   # MAVLink link and trim helpers of the sibling repository
from mpc import NOM, U_MIN, U_MAX, DT, DP, NSUB, RHO0, RHOC, trim, est_thrust

T_MAX_ROTOR = 30.0          # SIH_T_MAX: thrust of one rotor at full command [N]  (4 rotors = 120 N = 1.53 mg)
T_MAX_PUSHER = 60.0         # SIH standard VTOL: pusher thrust = 2 * SIH_T_MAX * u7

V_FILTER_TAU = 0.05                 # first-order filter on the airspeed measurement [s]
SIH_PARAMS = {              # plant of the SIL: the 8 kg quadplane of the article in PX4's simulation-in-hardware model
    "SIH_MASS": 8.0, "SIH_IXX": 0.9, "SIH_IYY": 1.2, "SIH_IZZ": 1.9, "SIH_IXZ": 0.0, "SIH_T_MAX": T_MAX_ROTOR, "SIH_Q_MAX": 2.5,
    "SIH_L_ROLL": 0.45, "SIH_L_PITCH": 0.45, "SIH_KDV": 0.0, "SIH_KDW": 0.8, "SIH_T_TAU": 0.15,
}
PX4_PARAMS = {              # autopilot configuration: pusher output driven by the offboard node, linear thrust curve, no RC failsafes in SITL
    "PWM_MAIN_FUNC8": 301, "THR_MDL_FAC": 0.0, "COM_RCL_EXCEPT": 4, "NAV_RCL_ACT": 0, "NAV_DLL_ACT": 0, "COM_OF_LOSS_T": 1.0,
    "COM_OBL_RC_ACT": 0, "MPC_THR_MIN": 0.02, "MC_PITCH_P": 5.0, "MC_PITCHRATE_P": 0.20, "MC_PITCHRATE_I": 0.10, "MC_PITCHRATE_D": 0.003,
    "MC_ROLLRATE_P": 0.20, "MC_ROLLRATE_I": 0.10, "MC_ROLLRATE_D": 0.003, "MC_YAWRATE_P": 0.3, "MC_YAWRATE_I": 0.15, "MC_YAW_P": 3.0, "MC_YAWRATE_MAX": 60.0,
    "VT_ARSP_TRANS": 25.0, "VT_F_TRANS_THR": 0.0, "VT_FWD_THRUST_EN": 0, "VT_ELEV_MC_LOCK": 0, "COM_DISARM_LAND": -1.0, "CA_FAILURE_MODE": 0,
}

def q_from_euler(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]

class PX4Link:
    """MAVLink link to the autopilot: measurements in, attitude/thrust setpoints and pusher command out"""
    def __init__(self, url="udpin:0.0.0.0:14540"):
        self.c = mavutil.mavlink_connection(url, source_system=1, source_component=191)   # MAV_COMP_ID_ONBOARD_COMPUTER
        print("waiting for PX4 heartbeat ..."); self.c.wait_heartbeat(); print(f"PX4 system {self.c.target_system} component {self.c.target_component}")
        self.att = None; self.lpos = None; self.hud = None; self.act = None; self.armed = False; self.mode = None; self.t_hb = time.time(); self.V_f = None
    def send_heartbeat(self):
        if time.time() - self.t_hb > 0.5:
            self.c.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0); self.t_hb = time.time()
    def set_param(self, name, value, timeout=3.0):
        """PX4 packs INT32 parameters bit-wise into the float field of PARAM_SET (MAVLink parameter protocol)"""
        if isinstance(value, int): ptype, fval = mavutil.mavlink.MAV_PARAM_TYPE_INT32, struct.unpack("<f", struct.pack("<i", value))[0]
        else: ptype, fval = mavutil.mavlink.MAV_PARAM_TYPE_REAL32, float(value)
        for _ in range(3):
            self.c.mav.param_set_send(self.c.target_system, self.c.target_component, name.encode(), fval, ptype)
            t0 = time.time()
            while time.time() - t0 < timeout:
                m = self.c.recv_match(type="PARAM_VALUE", blocking=True, timeout=timeout)
                if m is not None and m.param_id.strip("\x00") == name:
                    return struct.unpack("<i", struct.pack("<f", m.param_value))[0] if m.param_type == mavutil.mavlink.MAV_PARAM_TYPE_INT32 else m.param_value
        print(f"  ! param {name} not acknowledged"); return None
    def tick(self, timeout=5.0):
        """block until the next ATTITUDE message (100 Hz simulation time), keep the other measurements current; returns simulation time [s]"""
        while True:
            m = self.c.recv_match(blocking=True, timeout=timeout)
            if m is None: raise RuntimeError("no telemetry from the autopilot")
            t = m.get_type()
            if t == "LOCAL_POSITION_NED": self.lpos = m
            elif t == "VFR_HUD":
                self.hud = m; a_f = DT * 5 / (V_FILTER_TAU + DT * 5)     # VFR_HUD arrives at 50 Hz sim
                self.V_f = m.airspeed if self.V_f is None else self.V_f + a_f * (m.airspeed - self.V_f)
            elif t == "ACTUATOR_OUTPUT_STATUS": self.act = m
            elif t == "HEARTBEAT" and m.get_srcComponent() == 1:
                self.armed = bool(m.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED); self.mode = m.custom_mode
            elif t == "ATTITUDE": self.att = m; self.send_heartbeat(); return m.time_boot_ms * 1e-3
    def set_interval(self, msg_id, hz):
        self.c.mav.command_long_send(self.c.target_system, self.c.target_component, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, msg_id, 1e6 / hz, 0, 0, 0, 0, 0)
    def command(self, cmd, *params):
        p = list(params) + [0.0] * (7 - len(params)); self.c.mav.command_long_send(self.c.target_system, self.c.target_component, cmd, 0, *p)
    def pump(self):
        """drain the receive queue, keep the latest measurements"""
        while True:
            m = self.c.recv_match(blocking=False)
            if m is None: break
            t = m.get_type()
            if t == "ATTITUDE": self.att = m
            elif t == "LOCAL_POSITION_NED": self.lpos = m
            elif t == "VFR_HUD":
                self.hud = m; a_f = DT * 5 / (V_FILTER_TAU + DT * 5)     # VFR_HUD arrives at 50 Hz sim
                self.V_f = m.airspeed if self.V_f is None else self.V_f + a_f * (m.airspeed - self.V_f)
            elif t == "ACTUATOR_OUTPUT_STATUS": self.act = m
            elif t == "HEARTBEAT" and m.get_srcComponent() == 1:
                self.armed = bool(m.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED); self.mode = m.custom_mode
    def state(self):
        """(V, gamma, theta, altitude, vz) from the autopilot's estimates"""
        a, l, h = self.att, self.lpos, self.hud
        if a is None or l is None or h is None: return None
        vh = math.hypot(l.vx, l.vy); gam = math.atan2(-l.vz, max(vh, 0.3)); return float(self.V_f if self.V_f is not None else h.airspeed), gam, float(a.pitch), -float(l.z), -float(l.vz), float(a.yaw)
    def course_yaw(self, yaw_hold):
        """heading setpoint: the ground course above 3 m/s (coordinated flight), else the held heading"""
        l = self.lpos
        if l is None: return yaw_hold
        gs = math.hypot(l.vx, l.vy)
        if gs < 3.0: return yaw_hold
        return math.atan2(l.vy, l.vx)
    def send_attitude(self, pitch, thrust, yaw, roll=0.0):
        q = q_from_euler(roll, pitch, yaw)
        self.c.mav.set_attitude_target_send(int((time.time() * 1e3) % 4294967295), self.c.target_system, self.c.target_component, 0b00000111, q, 0, 0, 0, float(np.clip(thrust, 0.0, 1.0)))
    def send_pusher(self, value):
        self.command(187, float(np.clip(value, 0.0, 1.0)), 0, 0, 0, 0, 0, 0)
    def offboard_and_arm(self):
        self.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)   # PX4 custom main mode 6 = OFFBOARD
        time.sleep(0.2); self.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1, 21196)

def load_hot(tag="hot_v4"):
    RUNS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "runs"); cfg = json.load(open(_os.path.join(RUNS, f"{tag}_config.json")))
    net = N.build("hot", L=cfg["L"], d=cfg["d"], n_blocks=cfg["blocks"], heads=cfg.get("heads", 4)); net.load_state_dict(torch.load(_os.path.join(RUNS, f"{tag}_best.pt"))); return net
def run(mode, link, log_dir, u_init, xh_init, w_rho=None, t_hold=4.0, t_max=30.0, tag="hot_v4", use_filter=True):
    """front: hover-exit trim (6 m/s) -> cruise (15 m/s); back: cruise -> hover-exit trim, flown by HOT with the lift-safety module.
    Paced by the autopilot's 100 Hz attitude stream (simulation time); the command is updated every DP and held between updates."""
    rho_t = RHOC if mode == "front" else RHO0; rho_start = RHO0 if mode == "front" else RHOC
    ctrl = C.HOTController(load_hot(tag), use_filter=use_filter, reduced=True, n_samples=32)
    xh = xh_init.copy(); v0 = u_init.copy(); ok = True; log = dict(t=[], x=[], u=[], xh=[], ok=[], alt=[], t_upd=[], t_sim_upd=[], n_upd=0, interv=[])
    ctrl.reset(xh.copy(), v0.copy(), rho_t)
    yaw0 = link.state()[5]; t0 = link.tick(); k = 0; t_last = -1e9; gam_prev = None; gd_f = 0.0; V_prev = None; vd_f = 0.0
    print(f"  engage HOT from V={xh[0]:.2f} gamma={math.degrees(xh[1]):.2f} theta={math.degrees(xh[2]):.2f} That={xh[3]:.1f}/{xh[4]:.1f} N, hold input {np.round(v0, 2)}")
    while True:
        ts = link.tick(); t_now = ts - t0; st = link.state()
        V, gam, th, alt, vz, _ = st; xh[0], xh[1], xh[2] = V, gam, th
        if gam_prev is not None: gd_f = 0.8 * gd_f + 0.2 * (gam - gam_prev) / 0.01     # flight-path rate from the 100 Hz stream, first-order filtered
        if V_prev is not None: vd_f = 0.9 * vd_f + 0.1 * (V - V_prev) / 0.01              # acceleration from the filtered airspeed, first-order filtered
        V_prev = V
        gam_prev = gam
        if ts - t_last >= DP - 1e-3:
            t_last = ts; tw = time.perf_counter(); v0 = ctrl.step(xh.copy(), gd_f, rho_t, vd_f); log["t_upd"].append(time.perf_counter() - tw); log["n_upd"] += 1
            log["interv"].append(ctrl.F.stats["interventions"] if ctrl.F is not None else 0)
            log["t_sim_upd"].append(link.att.time_boot_ms * 1e-3 - ts)
        u = np.clip(v0, U_MIN, U_MAX)
        link.send_attitude(u[0], u[2] / (4 * T_MAX_ROTOR), link.course_yaw(yaw0))
        if k % 5 == 0: link.send_pusher(u[1] / T_MAX_PUSHER)
        log["t"].append(t_now); log["x"].append([V, gam, th]); log["u"].append(u.copy()); log["xh"].append(xh.copy()); log["ok"].append(ok); log["alt"].append(alt)
        xh = est_thrust(xh, u)
        k += 1
        if abs(V - rho_t) < 0.3 and log.get("t_done") is None: log["t_done"] = t_now
        if log.get("t_done") is not None and t_now - log["t_done"] > t_hold: break
        if t_now > t_max: print(f"  ! target not reached within {t_max:.0f} s"); break
    for key in ("t", "x", "u", "xh", "ok", "alt", "t_upd", "t_sim_upd", "interv"): log[key] = np.array(log[key])
    log["stats"] = dict(ctrl.F.stats) if ctrl.F is not None else {}
    return log
def fly_to_trim(link, rho, alt_ref, yaw0, duration=30.0):
    """bring the autopilot to a steady level flight at airspeed rho in multicopter mode with the trim command of the article
    (attitude setpoint + rotor collective + pusher), altitude held by a PI correction on the rotor collective, airspeed by the pusher.
    The thrust estimator (36) runs on the commands in force so that the MPC starts from consistent thrust states (Lemma 5)."""
    yaw0 = link.state()[5]
    xt, ut = trim(rho, NOM); t0 = link.tick(); ei = 0.0; ev = 0.0; k = 0; st = link.state(); xh = np.array([st[0], st[1], st[2], ut[1], ut[2]]); u = ut.copy()
    while True:
        ts = link.tick(); st = link.state(); V, gam, th, alt, vz, _ = st; e = alt_ref - alt; ei = float(np.clip(ei + e * DT, -30, 30)); ev = float(np.clip(ev + (rho - V) * DT, -8, 8))
        Tr = float(np.clip(ut[2] + 5.0 * e + 0.8 * ei - 8.0 * vz, 0, 4 * T_MAX_ROTOR)); Tc = float(np.clip(ut[1] + 1.5 * (rho - V) + 0.3 * ev, 0, 30))
        u = np.array([ut[0], Tc, Tr]); link.send_attitude(u[0], Tr / (4 * T_MAX_ROTOR), link.course_yaw(yaw0))
        if k % 5 == 0: link.send_pusher(Tc / T_MAX_PUSHER)
        xh[:3] = V, gam, th; xh = est_thrust(xh, u); k += 1
        if ts - t0 > duration: break
    st = link.state(); print(f"  trim {rho:.0f} m/s: V={st[0]:.2f} gamma={math.degrees(st[1]):.2f} deg theta={math.degrees(st[2]):.2f} deg alt={st[3]:.1f} m, commands Tc={u[1]:.1f} N Tr={u[2]:.1f} N, estimates {xh[3]:.1f}/{xh[4]:.1f} N")
    return u, xh

def takeoff(link, alt_ref, yaw0):
    """vertical climb on the rotors to alt_ref; the pitch setpoint cancels the forward ground speed that the cambered wing
    produces in the vertical airflow of the climb"""
    t0 = link.tick(); k = 0
    while True:
        ts = link.tick(); st = link.state(); V, gam, th, alt, vz, yaw = st; vz_ref = float(np.clip(0.6 * (alt_ref - alt), -2.0, 2.5)); thr = (8.0 * 9.81 + 12.0 * (vz_ref - vz)) / (4 * T_MAX_ROTOR)
        l = link.lpos; v_fwd = l.vx * math.cos(yaw) + l.vy * math.sin(yaw); pitch_sp = float(np.clip(0.08 * v_fwd, -0.25, 0.25))
        link.send_attitude(pitch_sp, thr, yaw0)
        if k % 5 == 0: link.send_pusher(0.0)
        k += 1
        if abs(alt - alt_ref) < 1.0 and abs(vz) < 0.3 and abs(v_fwd) < 0.5 and ts - t0 > 5: break
        if ts - t0 > 90: print("  ! takeoff timeout"); break

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["front", "back", "both"]); ap.add_argument("--log", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sil")); ap.add_argument("--url", default="udpin:0.0.0.0:14540")
    a = ap.parse_args(); os.makedirs(a.log, exist_ok=True); link = PX4Link(a.url)
    for name, val in {**SIH_PARAMS, **PX4_PARAMS}.items(): r = link.set_param(name, val); print(f"  {name} = {r}")
    for mid, hz in ((mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 100), (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 100), (mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, 50), (mavutil.mavlink.MAVLINK_MSG_ID_ACTUATOR_OUTPUT_STATUS, 20)): link.set_interval(mid, hz)
    print("waiting for estimates ..."); t0 = time.time()
    while link.state() is None: link.pump(); link.send_heartbeat(); time.sleep(0.05)
    yaw0 = link.state()[5]; print(f"EKF ready after {time.time()-t0:.1f} s, yaw {math.degrees(yaw0):.1f} deg")
    for _ in range(100): link.tick(); link.send_attitude(0.0, 0.0, yaw0)     # setpoint stream before offboard (1 s of simulation time)
    link.offboard_and_arm(); t0 = time.time()
    while not link.armed and time.time() - t0 < 30: link.tick(); link.send_attitude(0.0, 0.0, yaw0)
    print(f"armed={link.armed} mode={link.mode}")
    print("takeoff ..."); takeoff(link, 50.0, yaw0)
    modes = ["front", "back"] if a.mode == "both" else [a.mode]; logs = {}
    if modes[0] == "front": u0, xh0 = fly_to_trim(link, RHO0, 50.0, yaw0)
    else: fly_to_trim(link, RHO0, 50.0, yaw0); u0, xh0 = fly_to_trim(link, RHOC, 50.0, yaw0, 45.0)
    for mode in modes:
        print(f"== HOT {mode} transition"); lg = run(mode, link, a.log, u0, xh0); logs[mode] = lg
        tu = lg["t_upd"] * 1e3; tsu = lg["t_sim_upd"] * 1e3; print(f"  done: t_target={lg.get('t_done')} s, updates={lg['n_upd']}, wall update time mean {tu.mean():.1f} ms max {tu.max():.1f} ms, sim-time latency mean {tsu.mean():.1f} ms, stats={lg['stats']}")
        np.savez(os.path.join(a.log, f"sil_{mode}.npz"), **{k: v for k, v in lg.items() if isinstance(v, np.ndarray)})
        json.dump(dict(t_done=lg.get("t_done"), n_upd=lg["n_upd"], upd_ms_mean=float(tu.mean()), upd_ms_max=float(tu.max()), sim_latency_ms_mean=float(tsu.mean()), stats=lg["stats"]), open(os.path.join(a.log, f"sil_{mode}.json"), "w"), indent=1)
        if mode == "front" and len(modes) > 1: u0, xh0 = fly_to_trim(link, RHOC, 50.0, yaw0, 15.0)
    print("landing (disarm) ..."); link.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 21196)

if __name__ == "__main__": main()
