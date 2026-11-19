#!/usr/bin/env python3
"""Run an N0-VTLA fuse model (Sparsh tactile encoder + per-pad plan expectation, in-chunk correction and C-fresh;
N0-VTLA-fuse branch) on the real ALOHA robot (client).

Server (GPU box, N0-VTLA-fuse worktree; the ZMQ server, not the websocket serve_policy.py):
    VTLA_ASSET_ID=<asset> python scripts/serve_zmq.py --config real_joint_fuse \\
        --ckpt checkpoints/real_joint_fuse/<exp>/20000 --addr tcp://*:5557
    Direction 2 C-fresh (optional): also set  VTLA_DIR2_SURPRISE=<surprise_z.json> VTLA_DIR2_C_FRESH=1
    In-chunk correction (--refine): also set  VTLA_REFINE_CKPT=<dir with refine.safetensors> VTLA_DIR2_C_FRESH=1
        VTLA_DIR2_SURPRISE=<surprise_pad.json>   (the refine replies carry the per-pad C-fresh decision)
Client (this box, the robot's ROS env; needs pyzmq + msgpack):
    roslaunch aloha 4arms_tactile_teleop.launch
    python3 aloha_scripts/run_vtla_fused.py --host 10.240.77.20 --port 5557 --data hdf5 \\
        --prompt "plug the adapter into the socket" [--probe-every 5] [--takeover-after 3]  |  [--refine]

Differences from run_vtla_policy.py (its robot helpers are imported unchanged; that script is not modified):
  * ZMQ client (vtla_zmq_client.py), joint action space only: the server returns absolute joint targets (H, 14),
    clamped per arm and commanded exactly as run_vtla_policy.py's joint path;
  * the tactile baseline is the server's: reset() right after homing, the first predict of the episode captures it;
  * every control step's 4 tactile frames go into a ring buffer (vtla_fused_runtime.TactileHistory); each predict and
    each probe also sends the frame --hist steps earlier (Direction 3; the first frame while fewer steps passed);
  * --probe-every N > 0: Direction 2 C-fresh probes run in a background thread every N steps (the robot never waits
    for them); a "fresh" reply ends the chunk at the next step and the next predict starts from the live observation;
  * --takeover-after M > 0: M "fresh" events within --takeover-window steps -> the arms hold and the operator decides
    (RIGHT = resume the policy, LEFT / r = abort this round, ESC = quit); probes only;
  * --refine (default off; replaces --probe-every): each chunk starts with ``prepare`` (a predict that also builds the
    server's correction cache) and, while it runs, a background thread sends ``refine`` requests (the rows commanded so
    far, the live touch, the current state; at most one in flight, none in the last --refine-stop steps). A reply for
    "m steps executed" is written back from the first row not yet commanded (m + d, d = steps executed while it was in
    flight) as ORIGINAL plan + correction (corrections never accumulate, commanded rows never change; optional
    per-element caps --refine-max-delta / --refine-max-grip-delta). The same reply carries the per-pad C-fresh
    decision: "fresh" ends the chunk at the next step (no operator prompt on this path);
  * --data is REQUIRED: the recorder FORMAT of the checkpoint's training data, which fixes two things the model was
    trained with (--fps / --state-gripper override them):
        hdf5   the ALOHA HDF5 recorders (this repo's record_episodes_aloha2.py and hil_rollout.py --format hdf5;
               tactile_acuity's record_episodes.py, e.g. aloha_insertion_v1), converted by N0-VTLA-fuse
               convert_aloha_hdf5_canonical.py: 50 fps; gripper state = normalized FINGER position,
               PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_finger) as real_env.get_qpos (open ~0.27-0.29, closed ~-0.11)
        vtla   record_episodes_aloha2_vtla.py (CSV + mp4), converted by convert_canonical_data.py --robot aloha
               --aloha-action-space joint_position: 30 fps; gripper state = normalized gripper MOTOR angle
               (joint_state_vector, as run_vtla_policy.py; open ~1.0)
    The chunk is executed at --fps (a wrong rate plays the trajectory back at the wrong speed). The gripper STATE
    scales differ; the gripper ACTION is the master-normalized value in both;
  * --hist defaults to round(0.16 * fps), the server config's default k; the server log prints the k it serves
    ("history offsets") and the two must agree;
  * one JSONL log per episode in --log-dir: chunk starts, probe replies, takeover requests (--refine: refine replies
    with d / write-back stats, chunk ends with the executed rows, overruns and refine totals in the end record).
Colour order: cameras (usb_cam rgb8 passthrough) and tactile (Xense rectify bytes unchanged) are TRUE RGB, as in
run_vtla_policy.py and as the converted training data.
"""

import argparse
import collections
import json
import logging
import os
import time

import numpy as np

from constants import (PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_LEFT_GRIPPER_JOINT_CLOSE,
                       PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_CLOSE,
                       PUPPET_RIGHT_GRIPPER_JOINT_OPEN)
from one_side_teleop import KeyboardControl
from real_env import make_real_env
from record_episodes_aloha2_vtla import DEFAULT_TACTILE_MAP, TACTILE_SENSORS, video_sources
from run_vtla_policy import (CAM_MODEL_KEY, JOINT_LAYOUT, STALE_STREAM_S, RGBStreams, clamp_joint_target,
                             compute_observation_state, home_puppets, park_puppets_on_quit)
from tactile_stream import DEFAULT_MAX_TACTILE_AGE_S
from vtla_fused_runtime import REFINE_LOG_KEYS, AsyncProbe, AsyncRefine, ChunkPlan, TactileHistory, TakeoverMonitor
from vtla_zmq_client import MODEL_TACTILE_SIZE, VtlaZmqClient

HIST_SECONDS = 0.16   # the N0-VTLA real configs' default history span (config.py _HIST_SECONDS)
REFINE_STOP_SECONDS = 0.15   # --refine-stop before any d was measured: no new refine this close to the chunk end
# the training data's recorder -> frame rate and gripper-state convention (module docstring, --data)
DATA_PRESETS = {
    "hdf5": {"fps": 50.0, "state_gripper": "finger"},
    "vtla": {"fps": 30.0, "state_gripper": "joint"},
}


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", required=True, help="policy server host, e.g. 10.240.77.20")
    p.add_argument("--port", type=int, default=5557)
    p.add_argument("--data", required=True, choices=sorted(DATA_PRESETS),
                   help="recorder format of the training data: sets --fps and --state-gripper (module docstring)")
    p.add_argument("--fps", type=float, default=None, help="execution rate = the training data's frame rate "
                   "(default from --data: 50 hdf5, 30 vtla)")
    p.add_argument("--state-gripper", "--state_gripper", dest="state_gripper", choices=("finger", "joint"),
                   default=None, help="gripper entries of the state (default from --data)")
    p.add_argument("--hist", type=int, default=None, help="history offset k in steps (default round(0.16 * fps))")
    p.add_argument("--prompt", default="Perform the task.")
    p.add_argument("--third_camera", choices=("cam_high", "cam_low"), default="cam_high")
    for slot, sensor in DEFAULT_TACTILE_MAP.items():
        p.add_argument("--" + slot, choices=TACTILE_SENSORS, default=sensor)
    p.add_argument("--action_horizon", type=int, default=50, help="steps of each returned chunk executed at most")
    p.add_argument("--max_episode_steps", type=int, default=2000)
    p.add_argument("--num_episodes", type=int, default=1)
    p.add_argument("--gripper_moving_time", type=float, default=None)
    p.add_argument("--stream_timeout", type=float, default=5.0)
    p.add_argument("--max_stream_age", type=float, default=STALE_STREAM_S)
    p.add_argument("--max_tactile_age", type=float, default=DEFAULT_MAX_TACTILE_AGE_S)
    p.add_argument("--codec", choices=("png", "jpg"), default="png", help="image encoding of the chunk predicts")
    p.add_argument("--probe-every", "--probe_every", dest="probe_every", type=int, default=0,
                   help="Direction 2 C-fresh probe every N steps (0 = off; needs a server with VTLA_DIR2_C_FRESH=1)")
    p.add_argument("--probe-codec", "--probe_codec", dest="probe_codec", choices=("png", "jpg"), default="jpg")
    p.add_argument("--probe-full-res", "--probe_full_res", dest="probe_full_res", action="store_true",
                   help="send NATIVE tactile frames with probes / refines (default: resized to the model size 224 on "
                        "the robot, same resize as the server; only needed for a server older than 2026-11-18)")
    p.add_argument("--takeover-after", "--takeover_after", dest="takeover_after", type=int, default=0,
                   help="ask the operator to take over after this many 'fresh' events in --takeover-window (0 = off)")
    p.add_argument("--takeover-window", "--takeover_window", dest="takeover_window", type=int, default=None,
                   help="window in steps (default 5 s worth)")
    p.add_argument("--refine", action="store_true",
                   help="in-chunk tactile correction (prepare / refine; needs a server with VTLA_REFINE_CKPT and "
                   "VTLA_DIR2_C_FRESH=1); replaces --probe-every")
    p.add_argument("--refine-codec", "--refine_codec", dest="refine_codec", choices=("png", "jpg"), default="jpg")
    p.add_argument("--refine-timeout", "--refine_timeout", dest="refine_timeout", type=float, default=2.0,
                   help="reply timeout of one refine (s); short, so a lost reply cannot hold the next prepare")
    p.add_argument("--refine-every", "--refine_every", dest="refine_every", type=int, default=1,
                   help="at least this many steps between refine requests (1 = as soon as the previous reply is in)")
    p.add_argument("--refine-max-delta", "--refine_max_delta", dest="refine_max_delta", type=float, default=0.0,
                   help="cap on |corrected - plan| per arm joint, rad (0 = only the server's bound)")
    p.add_argument("--refine-max-grip-delta", "--refine_max_grip_delta", dest="refine_max_grip_delta", type=float,
                   default=0.0, help="cap on |corrected - plan| per gripper action (0 = only the server's bound)")
    p.add_argument("--refine-stop", "--refine_stop", dest="refine_stop", type=int, default=None,
                   help="no new refine in the last N steps of a chunk (default adaptive: max recent d + 1, "
                   "round(0.15 * fps) before any d was measured)")
    p.add_argument("--timeout", type=float, default=30.0, help="server reply timeout (s)")
    p.add_argument("--log-dir", "--log_dir", dest="log_dir", default="logs/vtla_fused")
    return p


def resolve_args(args):
    """Fill --fps / --state-gripper from --data and --hist from --fps (in place); returns args."""
    preset = DATA_PRESETS[args.data]
    if args.fps is None:
        args.fps = preset["fps"]
    if args.state_gripper is None:
        args.state_gripper = preset["state_gripper"]
    if args.hist is None:
        args.hist = max(1, int(round(HIST_SECONDS * args.fps)))
    if args.fps <= 0 or args.hist < 1:
        raise SystemExit("--fps must be > 0 and --hist >= 1")
    if getattr(args, "refine", False):
        if args.probe_every > 0:
            raise SystemExit("--refine replies carry the C-fresh decision; drop --probe-every")
        if args.takeover_after > 0:
            raise SystemExit("--takeover-after applies to --probe-every only; the --refine path never prompts the "
                             "operator")
        if args.refine_timeout <= 0 or args.refine_every < 1:
            raise SystemExit("--refine-timeout must be > 0 and --refine-every >= 1")
        if args.refine_max_delta < 0 or args.refine_max_grip_delta < 0:
            raise SystemExit("--refine-max-delta / --refine-max-grip-delta must be >= 0")
        if args.refine_stop is not None and args.refine_stop < 0:
            raise SystemExit("--refine-stop must be >= 0")
    return args


def observation_state(state_gripper, left_js, right_js, left_names, right_names, calibration, models):
    """The 32-dim padded state: [0:14] joints as the training data stores them, zeros after.

    Arm joints are the raw puppet joint positions in both conventions; the gripper entries [6] / [13] are the
    normalized gripper JOINT ("joint", joint_state_vector) or the normalized left-FINGER position ("finger",
    real_env.get_qpos: the HDF5 recorders, e.g. record_episodes_aloha2.py and aloha_insertion_v1)."""
    state = compute_observation_state("joint", left_js, right_js, left_names, right_names, calibration, models)
    if state_gripper == "finger":
        for slot, js in ((6, left_js), (13, right_js)):
            finger = float(PUPPET_GRIPPER_POSITION_NORMALIZE_FN(dict(zip(js.name, js.position))["left_finger"]))
            if not np.isfinite(finger):
                raise ValueError("Non-finite finger feedback")
            state[slot] = finger
    return state


def split_frames(frames, sources, tactile_names):
    """snapshot() frames {source: RGB} -> ({model camera: RGB}, {tactile view: RGB})."""
    cams, tacs = {}, {}
    for key, source in sources.items():
        short = key[len("observation.image."):]
        if source in tactile_names:
            tacs[short] = frames[source]
        else:
            cams[CAM_MODEL_KEY[short]] = frames[source]
    return cams, tacs


class EpisodeLog:
    def __init__(self, log_dir, episode):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "episode_{:03d}_{}.jsonl".format(episode, time.strftime("%Y%m%d_%H%M%S")))
        self._f = open(self.path, "a")

    def __call__(self, record):
        record = dict(record, t=round(time.time(), 3))
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def command(env, row, moving_time, accel_time):
    """One (14,) absolute joint target row -> both arms (clamped, held if non-finite) + both grippers."""
    for bot, joints, name in ((env.puppet_bot_left, row[JOINT_LAYOUT["left_joints"]], "left"),
                              (env.puppet_bot_right, row[JOINT_LAYOUT["right_joints"]], "right")):
        if np.isfinite(joints).all():
            bot.arm.set_joint_positions(clamp_joint_target(bot, joints), moving_time=moving_time,
                                        accel_time=accel_time, blocking=False)
        else:
            logging.warning("%s arm: non-finite joint prediction this tick, holding position", name)
    env.set_gripper_pose(float(np.clip(row[JOINT_LAYOUT["left_gripper"]], 0.0, 1.0)),
                         float(np.clip(row[JOINT_LAYOUT["right_gripper"]], 0.0, 1.0)))


def poll(keyboard):
    for event in keyboard.poll():
        if event == "esc":
            return "quit"
        if event in ("left", "reset"):
            return "reset"
    return None


def request_takeover(keyboard, log, step):
    """The arms hold (no new commands); RIGHT -> 'resume', LEFT / r -> 'reset', ESC -> 'quit'."""
    log({"event": "takeover_request", "step": step})
    if not keyboard.enabled:
        logging.warning("takeover requested but there is no keyboard: aborting this round")
        return "reset"
    print("\a*** Tactile mismatch persists: TAKE OVER. RIGHT = resume the policy, LEFT/r = abort round, ESC = quit ***")
    keyboard.drain()
    while True:
        for event in keyboard.poll():
            if event == "right":
                log({"event": "takeover_resume", "step": step})
                return "resume"
            if event in ("left", "reset"):
                return "reset"
            if event == "esc":
                return "quit"
        time.sleep(0.02)


def run_episode(client, env, streams, sources, args, keyboard, log, clock=time.monotonic, sleep=time.sleep):
    """Returns "done", "quit" or "reset" (as run_vtla_policy.run_episode)."""
    dt = 1.0 / args.fps
    moving_time = args.gripper_moving_time or dt
    accel_time = min(moving_time / 2.0, dt)
    for bot in (env.puppet_bot_left, env.puppet_bot_right):
        bot.arm.set_trajectory_time(moving_time=moving_time, accel_time=accel_time)
    left_names = env.puppet_bot_left.arm.group_info.joint_names
    right_names = env.puppet_bot_right.arm.group_info.joint_names
    calibration = [(PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN),
                   (PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, PUPPET_RIGHT_GRIPPER_JOINT_OPEN)]
    models = (env.puppet_bot_left.arm.robot_des, env.puppet_bot_right.arm.robot_des)

    # T_0 is the server's: reset() now (right after homing, before any contact); the first predict captures it from
    # a tactile frame delivered after this point.
    client.reset()
    if hasattr(streams, "wait_fresh_tactile"):
        streams.wait_fresh_tactile(time.time())
    history = TactileHistory(maxlen=2 * args.hist + 8)
    tac_size = None if getattr(args, "probe_full_res", False) else MODEL_TACTILE_SIZE
    probe = (AsyncProbe(client, args.probe_every, args.probe_codec, tactile_size=tac_size)
             if args.probe_every > 0 else None)
    refine = (AsyncRefine(client, args.refine_codec, args.refine_timeout, tactile_size=tac_size)
              if getattr(args, "refine", False) else None)
    window = args.takeover_window or int(round(5 * args.fps))
    monitor = TakeoverMonitor(args.takeover_after, window)

    def log_probes():                                           # replies arrive in the probe thread; logged here
        for chunk, k, reply in probe.drain():
            log(dict(reply, event="probe", chunk=chunk, k=k))

    # --refine bookkeeping (main thread only)
    cp = None                                                   # the running chunk (ChunkPlan)
    recent_d = collections.deque(maxlen=10)                     # measured d of the latest replies (adaptive stop)
    all_d = []
    tot = {"applied": 0, "overruns": 0, "max_overrun_ms": 0.0}

    def refine_stop():
        if args.refine_stop is not None:
            return args.refine_stop
        if recent_d:
            return max(recent_d) + 1
        return max(1, int(round(REFINE_STOP_SECONDS * args.fps)))

    def handle_refines(h, step, apply=True):
        """Apply (main thread, before the next command) and log the refine replies received so far."""
        for chunk_id, pid, m, r in refine.drain():
            if "error" in r:
                info = {"skipped": "error", "error": r["error"]}
            elif r.get("plan") is not None and (int(r["plan"]) != pid or int(r.get("m", m)) != m):
                info = {"skipped": "mismatched reply"}
            elif r.get("stale"):
                info = {"skipped": "stale on server"}
            elif r.get("fresh"):
                info = {"skipped": "fresh"}                     # ends the chunk; it carries no actions
            elif cp is None or pid != cp.plan_id:
                info = {"skipped": "stale plan"}
            elif not apply:
                info = {"skipped": "chunk ended"}
            else:
                info = cp.apply(pid, m, r.get("actions"))
                if "start" in info:
                    tot["applied"] += 1
            if "d" in info:
                recent_d.append(info["d"])
                all_d.append(info["d"])
            rec = dict(info, event="refine", chunk=chunk_id, plan=pid, m=m, h=h, step=step, rtt_ms=r.get("rtt_ms"))
            rec.update({key: r.get(key) for key in REFINE_LOG_KEYS if key in r})
            log(rec)

    def end_chunk(chunk_id, sent_before):
        if cp is None:
            return
        log({"event": "chunk_end", "chunk": chunk_id, "plan": cp.plan_id, "executed_steps": cp.cursor,
             "n_refine": refine.sent - sent_before, "n_applied": cp.n_applied,
             "d_max": max(cp.ds) if cp.ds else None,
             "executed_rows": np.round(cp.executed(), 4).tolist()})

    keyboard.drain()
    step, chunks, fresh_events = 0, 0, 0
    outcome = "done"
    while step < args.max_episode_steps and outcome == "done":
        ev = poll(keyboard)
        if ev:
            outcome = ev
            break
        frames = streams.snapshot(args.max_stream_age)          # chunk start: all fresh, or raise
        cams, tacs = split_frames(frames, sources, streams.tactile_names)
        history.push(step, tacs)
        state = observation_state(args.state_gripper, env.puppet_bot_left.dxl.joint_states,
                                  env.puppet_bot_right.dxl.joint_states, left_names, right_names, calibration, models)
        t0 = clock()
        if refine is None:
            actions = client.predict(state, args.prompt, cams, tacs, {args.hist: history.frame(step - args.hist)},
                                     codec=args.codec)
            log({"event": "chunk", "step": step, "chunk": chunks, "rtt_ms": round((clock() - t0) * 1000, 1),
                 "infer_ms": client.last_infer_time_ms})
        else:
            actions, plan_id = client.prepare(state, args.prompt, cams, tacs,
                                              {args.hist: history.frame(step - args.hist)}, codec=args.codec)
            rtt_ms = round((clock() - t0) * 1000, 1)
            refine.start_chunk(plan_id)
            handle_refines(-1, step, apply=False)               # the previous chunk's last replies: never applied
            cp = ChunkPlan(plan_id, actions, min(args.action_horizon, len(actions)),
                           args.refine_max_delta, args.refine_max_grip_delta)
            sent_before, last_sent_m = refine.sent, 0
            log({"event": "chunk", "step": step, "chunk": chunks, "rtt_ms": rtt_ms,
                 "infer_ms": client.last_infer_time_ms, "plan": plan_id,
                 "plan_rows": np.round(cp.plan[:cp.n_exec], 4).tolist()})
        chunks += 1
        if probe is not None:
            probe.start_chunk()
            log_probes()                                        # the previous chunk's last replies
        for h in range(min(args.action_horizon, len(actions))):
            ev = poll(keyboard)
            if ev:
                outcome = ev
                break
            tick = clock()                                      # the drain/apply/log below counts against dt
            if refine is not None:
                handle_refines(h, step)                         # corrections land before row h is commanded
                if refine.fresh():                              # a fresh reply already read: no more rows of this plan
                    fresh_events += 1
                    log({"event": "fresh", "step": step, "chunk_step": h, "source": "refine"})
                    break
            command(env, cp.take(h) if refine is not None else actions[h], moving_time, accel_time)
            step += 1
            if step >= args.max_episode_steps:
                break
            fresh_frames = True
            try:                                                # this step's tactile: history buffer + probes
                frames = streams.snapshot(args.max_stream_age)
                cams, tacs = split_frames(frames, sources, streams.tactile_names)
                history.push(step, tacs)
            except RuntimeError as exc:
                fresh_frames = False
                logging.warning("step %d: no fresh frames (%s); no history frame / probe this step", step, exc)
            if probe is not None:
                log_probes()
                if probe.fresh():                               # tactile mismatch: end the chunk here
                    fresh_events += 1
                    log({"event": "fresh", "step": step, "chunk_step": h + 1})
                    if monitor.event(step):
                        decision = request_takeover(keyboard, log, step)
                        if decision != "resume":
                            outcome = decision
                        monitor.clear()
                    break
                if fresh_frames:
                    probe.maybe_probe(h + 1, h + 1, tacs, {args.hist: history.frame(step - args.hist)})
            if refine is not None:
                if refine.fresh():                              # fresh reply during command/snapshot: end here
                    handle_refines(h + 1, step, apply=False)
                    fresh_events += 1
                    log({"event": "fresh", "step": step, "chunk_step": h + 1, "source": "refine"})
                    break
                m = h + 1                                       # commanded rows; this step's touch is after them
                if (fresh_frames and cp.n_exec - m > refine_stop() and m - last_sent_m >= args.refine_every
                        and not refine.busy()):
                    try:
                        state = observation_state(args.state_gripper, env.puppet_bot_left.dxl.joint_states,
                                                  env.puppet_bot_right.dxl.joint_states, left_names, right_names,
                                                  calibration, models)
                    except ValueError as exc:
                        logging.warning("step %d: no refine (%s)", step, exc)
                    else:
                        if refine.maybe_refine(m, cp.executed(), tacs, {args.hist: history.frame(step - args.hist)},
                                               state=state):
                            last_sent_m = m
            elapsed = clock() - tick
            if refine is not None and elapsed > dt:
                tot["overruns"] += 1
                tot["max_overrun_ms"] = max(tot["max_overrun_ms"], round((elapsed - dt) * 1000, 1))
            sleep(max(0.0, dt - elapsed))
        if refine is not None:
            end_chunk(chunks - 1, sent_before)
    if probe is not None:
        probe.finish()
        log_probes()
    end = {"event": "end", "outcome": outcome, "steps": step, "chunks": chunks, "fresh": fresh_events,
           "probe_errors": probe.errors if probe is not None else 0}
    if refine is not None:
        refine.finish()
        handle_refines(-1, step, apply=False)                   # in flight at the end: logged, never applied
        end.update(refines=refine.sent, refine_applied=tot["applied"], refine_errors=refine.errors,
                   d_mean=round(float(np.mean(all_d)), 3) if all_d else None,
                   d_max=int(max(all_d)) if all_d else None,
                   overruns=tot["overruns"], max_overrun_ms=tot["max_overrun_ms"])
    log(end)
    return outcome


def main():
    logging.basicConfig(level=logging.INFO)
    args = resolve_args(build_arg_parser().parse_args())
    logging.info("data %s: fps %.1f (execution rate), gripper state %s, history offset k = %d steps (the server log "
                 "must show the same k)", args.data, args.fps, args.state_gripper, args.hist)
    if args.refine:
        logging.info("refine on: codec %s, timeout %.2f s, every %d, stop %s, max delta arm %g / gripper %g (0 = "
                     "server bound only)", args.refine_codec, args.refine_timeout, args.refine_every,
                     "adaptive" if args.refine_stop is None else args.refine_stop, args.refine_max_delta,
                     args.refine_max_grip_delta)
    tactile_map = {slot: getattr(args, slot) for slot in DEFAULT_TACTILE_MAP}
    sources = video_sources(args.third_camera, tactile_map)
    client = VtlaZmqClient(args.host, args.port, timeout_s=args.timeout)
    client.reset()        # fail here, before the arms move, if the server is not up (reset is harmless)
    logging.info("Policy server %s answered.", client.addr)
    env = make_real_env(init_node=True, setup_robots=True)
    streams = RGBStreams(sources, tactile_map, max_tactile_age=args.max_tactile_age)
    keyboard = KeyboardControl()
    try:
        logging.info("Waiting for camera/tactile streams...")
        streams.wait_ready(args.stream_timeout, args.max_stream_age)
        episode = 0
        while episode < args.num_episodes:
            logging.info("Homing both puppet arms to START_ARM_POSE...")
            home_puppets(env)
            if not keyboard.enabled:
                logging.info("Homed. No tty on stdin: starting immediately.")
            else:
                print("Homed. Press the RIGHT arrow key to start (ESC to quit).")
                keyboard.drain()
                started = False
                while not started:
                    for event in keyboard.poll():
                        if event == "right":
                            started = True
                        elif event == "esc":
                            park_puppets_on_quit()
                            return
                    time.sleep(0.02)
                print("LEFT arrow or r = abort this round and re-home, ESC/q = stop and power down.")
            log = EpisodeLog(args.log_dir, episode)
            try:
                outcome = run_episode(client, env, streams, sources, args, keyboard, log)
            finally:
                log.close()
            logging.info("Episode %d/%d: %s (log %s).", episode + 1, args.num_episodes, outcome, log.path)
            if outcome == "quit":
                park_puppets_on_quit()
                break
            if outcome == "reset":
                print("Reset: re-homing and retrying this round...")
                continue
            episode += 1
    finally:
        streams.close()
        keyboard.close()
        client.close()


if __name__ == "__main__":
    main()
