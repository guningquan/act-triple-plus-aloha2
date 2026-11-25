#!/usr/bin/env python3
"""Run a ForceGuide campaign on the real robot: the P0 and its guided conditions, all served by serve_forceguide.py.

serve_forceguide (forceguide repo) serves one frozen pi0.5 P0 and several conditions (CPM + serving flags) in one
process; each trial's reset message selects its condition, so conditions can be interleaved trial by trial without a
server restart. This script is the robot side: the same observation pipeline for every condition (cameras, state,
all campaign tactile sensors), blocking chunk-start plans padded to one fixed pause, in-chunk refreshes for refresh
conditions (forceguide_client.RefreshingChunkExecutor), the trial schedule, the operator's outcome entry and the
logs the analysis needs. Every real-robot condition, P0 included, must go through serve_forceguide.

Server (forceguide repo; recommended on aloha's 3090, localhost):
    uv run scripts/forceguide/serve_forceguide.py --port 8000 --policy-config pi05_board_insertion_tactile \
        --policy-dir <P0 checkpoint> --norm-stats <dir> --conditions conditions.json --log-dir <dir>
Client (this machine, `conda activate aloha`, after `roslaunch aloha 4arms_tactile_teleop.launch`, publisher
image_codec raw):
  1. schedule (once, no robot): blocks of one initial condition, conditions in random order inside each block
        python3 aloha_scripts/run_forceguide_policy.py --make_schedule trials.jsonl \
            --schedule_conditions p0 tg_real --schedule_inits 0-49 --schedule_seed 0
  2. timing preflight (robot static, XENSE publisher running): RTT of plans and refreshes with the real payload,
     the lead the refresh conditions need and the chunk-start pause; refuses (exit 2) if --lead is too small or a
     refresh gets an error reply. Latency does not depend on eta, so this may run against the conditions served
     WITHOUT an eta calibration (nominal eta). Without --lead it measures with the conditions' calibrated lead, if
     the server reports one, else max(3, min_lead)
        python3 aloha_scripts/run_forceguide_policy.py --host 127.0.0.1 --schedule trials.jsonl \
            --log_dir forceguide_logs/campaign1 --preflight 20
  3. choose the campaign lead L >= the preflight's required_lead, calibrate eta for exactly that lead (forceguide
     repo: offline_guidance_check.py --lead L writes calibration.json) and restart serve_forceguide with the
     calibrated conditions: an eta is valid only for the lead it was calibrated with, the server refuses refreshes
     with another lead, and this client refuses a --lead that differs from the served calibration (unless --force)
  4. trials (re-running the same command resumes; infrastructure failures are re-run automatically). A --schedule
     run needs an explicit --log_dir, a recent preflight report (--preflight_report, default <log_dir>/preflight.json,
     at most --preflight_max_age_h old) and a chunk-start pause > 0 (--plan_pause_s, default the preflight's
     recommendation), unless --force; the pause and the lead must stay the same when a campaign is resumed
        python3 aloha_scripts/run_forceguide_policy.py --host 127.0.0.1 --schedule trials.jsonl --lead L \
            --log_dir forceguide_logs/campaign1
A run without a schedule file (--condition NAME --num_episodes N: trials r0000.. with initial condition = noise id =
index) carries a run label, --run_label or by default the name of its log dir. The label is part of its trial
identity (schedule_sha256) and of every episodes.jsonl row, so runs in different log dirs never merge in the
analysis: it reports them as <condition>@<label> and pairs two labelled runs trial by trial (same initial condition
and noise index; analyze_robot_trials --pairs p0@p0_pilot=p0@p0_conv_realenv). Resuming in the same log dir keeps
the label; another label there is refused like another schedule (unless --force).
The operator only sees the trial id and the initial condition, never the condition name.

Keys (terminal):  RIGHT start the trial; while running RIGHT = end it (then enter the outcome)
                  LEFT or r abort (followers stop where they are; then enter the outcome or choose re-run)
                  ESC or q  stop the session (grippers open, followers to the sleep pose, torque off)
A plan error, a stale camera or tactile sensor, or a changed server stops the trial as an infrastructure failure
(re-run), as does a trial with more than --max_bad_refresh_frac of its refresh slots lost to infrastructure faults
(failed requests, error replies, stale tactile data). Rejected, late, dropped and skipped-in-flight refreshes are
outcomes of the method and of its compute time: they are logged as protocol_flags and never re-run a trial.
Intention to treat: the operator enters the outcome of every attempt that ran (also of an infrastructure failure,
whose "stopped" line names the fault, before being told it will be re-run); episodes.jsonl keeps it next to the
status. The analysis takes a trial's re-run in place of an infrastructure failure and uses these answers in a
sensitivity analysis (analyze_robot_trials).

Timing: control loop at constants.DT (50 Hz) on a policy-step clock (0.6 policy steps per tick for a 30 Hz P0).
After each plan the first command waits until --plan_pause_s after the blocking tick began (every condition), and
after any overrun the loop sleeps one control period before the next command. Followers: FollowerCommander (joint
limit clamp, --max_joint_step per tick; gripper clipped to [0, 1] for VTLA-trained P0s).

Logs in --log_dir: session.json (handshake metadata, settings), episodes.jsonl (one line per trial attempt with the
operator outcome, refresh outcome counts, latencies, fingerprints), trials/<trial>_a<attempt>/ticks.jsonl (every
tick), requests.jsonl (every request; join with the server log on (session, req)), refresh.npz (old/new plans of
every refresh reply), warmup.jsonl, preflight.json. A trial folder that already holds data (a crashed attempt
without its episodes.jsonl row, another run forced into the log dir) is moved aside to
trials/<trial>_a<attempt>.prev<k>, never overwritten. client.log: the full log of every session in the log dir
(DEBUG, timestamps, threads, tracebacks); the console shows only the operator-facing lines (ClientLog: no record
that could reveal a trial's condition, no traceback).
"""

import argparse
import contextlib
import datetime
import json
import logging
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

from chunk_executor import PolicyError
from forceguide_client import (DEBUG_LOG, FILE_ONLY, INFRA_REFRESH_STATUSES, LATE_FLAG_FRAC, PROTOCOL_REFRESH_STATUSES,
                               CampaignConfig, ForceGuideAdapter, ForceGuideClient, ForceGuideConnection,
                               InfrastructureFailure, NoiseSchedule, RefreshingChunkExecutor, ServerInfo, SpliceGuard,
                               TactileRig, TrialContext, check_gripper_calibration, console_text, file_sha256,
                               gripper_calibration_from_metadata, load_schedule, make_schedule, required_lead,
                               rows_sha256, save_sidecar, summarize_ms, ticks_per_chunk, to_jsonable, write_jsonl,
                               write_schedule)
from run_pi_policy import StaleImage, capture, home_followers, image_ages, wait_for_start

FINAL_STATUSES = ("success", "failure")
STAGES = ("none", "grasp", "align", "insert", "done")
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    server = parser.add_argument_group("server")
    server.add_argument("--host", default=None, help="serve_forceguide host (127.0.0.1 when it runs on aloha)")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--prompt", default=None, help="Task prompt; omitted -> the training config's default prompt")
    server.add_argument("--infer_timeout", type=float, default=10.0,
                        help="Hard reply timeout of every request (s); a plan past it stops the trial")
    server.add_argument("--warmup_timeout", type=float, default=900.0, help="Reply timeout during the warm-up (s)")
    server.add_argument("--connect_timeout", type=float, default=30.0, help="Connect budget of the plan path (s)")
    trials = parser.add_argument_group("trials")
    trials.add_argument("--schedule", default=None, help="trials.jsonl: trial_id, init_condition_id, condition")
    trials.add_argument("--condition", default=None, help="Without --schedule: run this condition only")
    trials.add_argument("--num_episodes", type=int, default=1, help="Without --schedule: number of trials")
    trials.add_argument("--log_dir", default=None, help="Default: forceguide_logs/<date>_<time>")
    trials.add_argument("--run_label", default=None,
                        help="Without --schedule: label of this run, part of its trial identity and of every "
                             "episodes.jsonl row (letters, digits, ._+-; default: the log dir's name). The analysis "
                             "reports the run as <condition>@<label>; use a label once per run analysed together")
    trials.add_argument("--unblinded", action="store_true",
                        help="No operator blinding: the console names each trial's condition, shows the full reason of "
                             "an infrastructure failure and keeps log records during a trial (episodes.jsonl and "
                             "client.log are the same either way)")
    trials.add_argument("--no_outcome", action="store_true",
                        help="Do not ask for an outcome after a trial (the operator keeps their own record): an "
                             "attempt that ran is logged as unlabelled and counts as done, also when the run is "
                             "resumed; infrastructure failures are still re-run")
    trials.add_argument("--seed", type=int, default=0, help="Noise seed: default_rng([seed, noise_id, plan_index]); "
                        "a resumed campaign keeps its seed")
    trials.add_argument("--max_episode_steps", type=int, default=3000, help="Control ticks per trial (50 Hz)")
    trials.add_argument("--max_attempts", type=int, default=3, help="Attempts per trial (re-runs included)")
    trials.add_argument("--max_bad_refresh_frac", type=float, default=0.1,
                        help="More refresh slots lost to infrastructure faults (failed / error / skipped-stale / "
                             "skipped-disabled / malformed) than this -> infrastructure failure (re-run); more "
                             "rejected / late / dropped / skipped-in-flight slots only set a protocol flag")
    timing = parser.add_argument_group("timing and refresh")
    timing.add_argument("--lead", type=int, default=None,
                        help="k_pin = k + lead policy steps (refresh conditions): >= the preflight's required lead "
                             "and equal to the lead the served eta was calibrated with")
    timing.add_argument("--plan_pause_s", type=float, default=None,
                        help="Chunk-start pause, same for every condition (default: the preflight's p99 plan RTT)")
    timing.add_argument("--refresh_timeout", type=float, default=0.25,
                        help="Client-side apply deadline of a refresh reply, from its submission (s)")
    timing.add_argument("--action_horizon", type=int, default=None, help="Chunk steps executed (default: server's H)")
    timing.add_argument("--policy_hz", type=float, default=None, help="Must equal the server's unless --force")
    timing.add_argument("--refresh_every", type=int, default=None, help="Must equal the server's unless --force")
    timing.add_argument("--preflight", type=int, default=0, metavar="N",
                        help="Robot static: N chunks per condition, report RTT/lead/pause, write preflight.json, exit")
    timing.add_argument("--preflight_report", default=None, help="Default: <log_dir>/preflight.json")
    timing.add_argument("--preflight_max_age_h", type=float, default=24.0,
                        help="A --schedule run needs a preflight report at most this old (unless --force)")
    timing.add_argument("--no_warmup", action="store_true", help="Skip the warm-up chunk of every condition")
    safety = parser.add_argument_group("safety")
    safety.add_argument("--max_joint_step", type=float, default=0.1, help="Per-tick arm joint change limit (rad)")
    safety.add_argument("--max_stream_age", type=float, default=0.5, help="Stop the trial on older camera images")
    safety.add_argument("--max_tactile_age", type=float, default=0.2,
                        help="Tactile data older (or seq frozen longer) than this: skip refreshes, stop at a plan")
    safety.add_argument("--max_splice_jump", type=float, default=0.075, help="Refresh splice discontinuity (rad)")
    safety.add_argument("--max_plan_step", type=float, default=0.2, help="Largest step of a spliced plan (rad)")
    safety.add_argument("--max_gripper_jump", type=float, default=0.25, help="Gripper change at the splice point")
    safety.add_argument("--force", action="store_true", help="Allow --policy_hz/--refresh_every overrides, "
                        "a jpeg tactile publisher, a too small --lead or one that differs from the served eta "
                        "calibration, legacy 'pad' tactile resize, a campaign without a recent preflight report or "
                        "with an unpadded chunk-start pause, a resumed campaign with another pause / lead / seed / "
                        "schedule")
    tactile = parser.add_argument_group("tactile")
    tactile.add_argument("--tactile_sensor", action="append", default=[], metavar="KEY=SENSOR",
                         help="Sensor of a tactile key when its suffix is not the sensor name")
    tactile.add_argument("--no_tactile_calibration", action="store_true",
                         help="Do not re-zero the tactile sensors (/tactile/calibrate) before each trial")
    tactile.add_argument("--dataset_metadata", default=None,
                         help="Raw episode metadata.json whose gripper_calibration the VTLA state must match")
    tactile.add_argument("--tactile_wait", type=float, default=10.0, help="Wait for the first tactile frames (s)")
    schedule = parser.add_argument_group("schedule generation (no robot)")
    schedule.add_argument("--make_schedule", default=None, metavar="OUT", help="Write a trials.jsonl and exit")
    schedule.add_argument("--schedule_conditions", nargs="+", default=None)
    schedule.add_argument("--schedule_inits", nargs="+", default=None, help="Initial condition ids, e.g. 0-49")
    schedule.add_argument("--schedule_seed", type=int, default=0)
    schedule.add_argument("--schedule_repeats", type=int, default=1)
    return parser


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------

class NullKeyboard:
    enabled = False

    def poll(self):
        return []

    def drain(self):
        pass

    def close(self):
        pass


class DryFollower:
    """FollowerCommander stand-in while nothing may move (warm-up, preflight): commands are only recorded."""

    def __init__(self, measured_fn=None):
        self.last = None
        self._measured = measured_fn

    def send(self, command):
        self.last = np.asarray(command, dtype=np.float64).copy()
        return self.last.copy()

    def measured(self):
        return None if self._measured is None else self._measured()


def parse_inits(values):
    out = []
    for value in values or []:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            match = re.match(r"^(\d+)-(\d+)$", token)
            if match:
                out.extend(range(int(match.group(1)), int(match.group(2)) + 1))
            else:
                out.append(int(token))
    return out


def parse_sensor_overrides(values):
    overrides = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--tactile_sensor expects KEY=SENSOR, got {!r}".format(value))
        key, sensor = value.split("=", 1)
        overrides[key.strip()] = sensor.strip()
    return overrides


def key_outcome(events):
    for event in events:
        if event == "esc":
            return "quit"
        if event in ("left", "reset"):
            return "aborted"
        if event == "right":
            return "ended"
    return None


def clamp_flags(planned, sent, puppets, clip_gripper):
    """Which FollowerCommander safety layer changed this tick's command: joint limits, the per-tick step limit,
    the gripper clip (requested vs sent)."""
    planned = np.asarray(planned, dtype=np.float64)
    sent = np.asarray(sent, dtype=np.float64)
    limit, step = [], []
    for index, bot in enumerate(puppets):
        lower = np.asarray(bot.arm.group_info.joint_lower_limits[:6], dtype=np.float64)
        upper = np.asarray(bot.arm.group_info.joint_upper_limits[:6], dtype=np.float64)
        arm = planned[index * 7:index * 7 + 6]
        clipped = np.clip(arm, lower, upper)
        limit.append(bool((clipped != arm).any()))
        step.append(bool((np.abs(sent[index * 7:index * 7 + 6] - clipped) > 1e-9).any()))
    grippers = planned[[6, 13]]
    return {"limit": limit, "step": step,
            "gripper_clip": bool(clip_gripper and ((grippers < 0.0) | (grippers > 1.0)).any())}


def git_rev(path=REPO_DIR):
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=5)
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
        rev = out.stdout.decode().strip()
        return rev + ("-dirty" if dirty.stdout.strip() else "") if rev else None
    except Exception:  # noqa: BLE001
        return None


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


RUN_LABEL_PATTERN = re.compile(r"[\w.+-]+")


def run_label_for(log_dir, explicit=None):
    """The run label of a run without a schedule file (contract C9): --run_label, or the name of its log dir. The
    label is part of the trial identity (schedule_sha256) and names the run in the analysis (<condition>@<label>,
    --pairs a=b), hence letters, digits and _.+- only. ValueError otherwise."""
    if explicit is not None:
        if not RUN_LABEL_PATTERN.fullmatch(explicit):
            raise ValueError("--run_label {!r}: use letters, digits and _.+- only (the analysis names the run "
                             "<condition>@<label> and pairs runs as a=b)".format(explicit))
        return explicit
    label = os.path.basename(os.path.normpath(os.path.abspath(log_dir)))
    if not RUN_LABEL_PATTERN.fullmatch(label):
        raise ValueError("the log dir's name {!r} cannot be the run label (letters, digits and _.+- only): pass "
                         "--run_label".format(label))
    return label


def generated_rows(condition, num_episodes, run_label):
    """The trials of a run without a schedule file: r0000.. with block, initial condition and noise id = the index,
    and the run label, which enters the rows' sha256 so that runs in different log dirs never share trial
    identities while a resume in the same log dir keeps them."""
    return [{"trial_id": "r{:04d}".format(i), "block": i, "init_condition_id": i, "noise_id": i,
             "condition": condition, "run_label": run_label} for i in range(num_episodes)]


# --------------------------------------------------------------------------------------------
# One round
# --------------------------------------------------------------------------------------------

def run_forceguide_episode(executor, follower, keyboard, capture_frame, rig, max_steps, control_dt, plan_pause_s=0.0,
                           clock=time.monotonic, sleep=time.sleep, ctx=None, camera_ages=None, clamp_fn=None,
                           show_progress=False, label="", measured_fn=None):
    """One round. Returns (outcome, reason): "done" (max_steps), "ended" (RIGHT), "aborted" (LEFT / r), "quit"
    (ESC / q) or "infrastructure" (plan error, stale camera or tactile, changed server).

    Per tick: keys, tactile capture for the steps first reached at this position (before the command), the
    executor step (which may block on a plan), then for a plan the pause padding up to plan_pause_s after the tick
    began and a second key poll (an abort pressed during the plan must stop a'[0]), the command, and the tick log
    (with the measured follower pose from measured_fn at the tick start, for tracking error and jerk).
    After an overrun longer than one period the loop sleeps one control_dt before the next command, so a'[0] and
    a'[1] are never sent back to back, the same in every condition.
    """
    labels = dict(ctx.labels) if ctx is not None else {}
    bar = None
    if show_progress:
        from tqdm import tqdm

        bar = tqdm(total=max_steps, unit="tick", dynamic_ncols=True, desc=label)
    keyboard.drain()
    next_tick = clock()
    previous = None
    try:
        for tick in range(max_steps):
            tick_start = clock()
            period = None if previous is None else tick_start - previous
            previous = tick_start
            events = list(keyboard.poll())
            outcome = key_outcome(events)
            if outcome is not None:
                return outcome, "key"
            measured = None
            if measured_fn is not None:
                try:
                    measured = measured_fn()
                except Exception:  # noqa: BLE001 - logging only, never stops the round
                    measured = None
            tactile = rig.tick(executor.position) if rig is not None else {}
            try:
                action = executor.step(capture_frame)
            except (PolicyError, StaleImage) as exc:  # PolicyError includes InfrastructureFailure
                return "infrastructure", "{}: {}".format(type(exc).__name__, exc)
            except Exception as exc:  # noqa: BLE001 - a bug: stop autonomous motion, keep the traceback
                # client.log only: the frames and message can name the condition's code path; the console shows the
                # exception class (console_reason) and episodes.jsonl the reason
                logging.exception("%s stopped by an unexpected client error", label or "the trial", extra=FILE_ONLY)
                return "infrastructure", "{}: {}".format(type(exc).__name__, exc)
            info = dict(executor.last_tick)
            pause_ms = pause_overrun = None
            if info.get("planned"):
                remaining = tick_start + plan_pause_s - clock()
                if remaining > 0:
                    sleep(remaining)
                pause_ms = 1000.0 * (clock() - tick_start)
                pause_overrun = bool(plan_pause_s > 0 and remaining <= 0)
                if ctx is not None:
                    ctx.annotate_plan(info.get("plan_req"), pause_ms=pause_ms, pause_overrun=pause_overrun)
                more = list(keyboard.poll())
                events += more
                outcome = key_outcome(more)
                if outcome is not None:
                    return outcome, "key during the plan"
            sent = follower.send(action)
            executor.note_sent(sent)
            record = {"t": tick_start, "period": period, "tick": tick, "chunk": info.get("chunk"),
                      "position": info.get("position"), "offset": info.get("offset"), "step": info.get("step"),
                      "k": info.get("k"), "plan": bool(info.get("planned")), "src_req": info.get("src_req"),
                      "planned_action": np.asarray(action, dtype=np.float64), "sent_action": sent,
                      "tactile": tactile, "keys": events}
            record.update(labels)
            if measured is not None:
                record["measured"] = np.asarray(measured, dtype=np.float64)
            if pause_ms is not None:
                record.update(pause_ms=pause_ms, pause_overrun=pause_overrun)
            if clamp_fn is not None:
                record["clamp"] = clamp_fn(action, sent)
            if camera_ages is not None:
                record["camera_ages"] = camera_ages()
            if bar is not None:
                bar.update(1)
            next_tick += control_dt
            delay = next_tick - clock()
            record["overrun"] = bool(delay < 0)
            if ctx is not None:
                ctx.ticks.append(record)
            if delay > 0:
                sleep(delay)
            elif delay < -control_dt:
                sleep(control_dt)  # one full period after a blocking plan / long overrun, then a fresh schedule
                next_tick = clock()
        return "done", None
    finally:
        if bar is not None:
            bar.close()


def ask_outcome(input_fn=input, print_fn=print):
    """Operator outcome of one trial: success / failure (stage reached, failure type, notes) or re-run."""

    def ask(prompt, choices=None, allow_empty=True):
        while True:
            try:
                answer = input_fn(prompt)
            except EOFError:
                return None
            answer = (answer or "").strip()
            if choices is None or answer.lower() in choices or (allow_empty and not answer):
                return answer
            print_fn("Please answer one of: {}".format(", ".join(choices)))

    first = ask("Success? [y]es / [n]o / [r]e-run (setup problem, not counted): ",
                ("y", "yes", "n", "no", "r", "rerun", "re-run"), allow_empty=False)
    if first is None:
        return {"labelled": False, "success": None, "rerun": False, "stage": None, "failure_type": None, "notes": None}
    if first.lower() in ("r", "rerun", "re-run"):
        notes = ask("Why re-run? ")
        return {"labelled": True, "success": None, "rerun": True, "stage": None, "failure_type": None,
                "notes": notes}
    success = first.lower() in ("y", "yes")
    stage = ask("Stage reached [{}]: ".format("/".join(STAGES)), STAGES)
    failure_type = None if success else ask("Failure type (e.g. miss, jam, slip, collision, timeout, other): ")
    notes = ask("Notes: ")
    return {"labelled": True, "success": success, "rerun": False, "stage": stage or ("done" if success else None),
            "failure_type": failure_type, "notes": notes}


# What the blinded operator sees of an infrastructure failure: one fixed category per kind of fault. A reason can carry
# the server's own text (a refused plan names the condition and the tactile input it reads, a traceback names the
# guidance code) and the client's connection names ("refresh: ..."), any of which tells a guided, refresh or tactile
# condition from P0. Reasons are "<exception class>: <message>" (run_forceguide_episode, the reset in
# _run_trial_rows); episodes.jsonl keeps them in full.
CONSOLE_DETAILS = "; details in episodes.jsonl"
CONSOLE_BY_CLASS = {"ServerChanged": "the policy server changed",  # main or refresh connection, condition fingerprint
                    "ServerError": "the policy server raised an error",  # a traceback string (plan or reset)
                    "ConnectFailed": "connection to the policy server failed",
                    "ConnectionLost": "connection to the policy server failed",
                    "StaleTactile": "camera/tactile data stale",  # every condition uploads tactile (CampaignConfig)
                    "StaleImage": "camera/tactile data stale"}
CONSOLE_BY_MESSAGE = (  # (class, message prefix, category): forceguide_client ForceGuideClient.reset / plan
    ("InfrastructureFailure", "plan failed:", "plan failed (server error)"),
    ("InfrastructureFailure", "reset failed:", "reset failed (server error)"),
    ("InfrastructureFailure", "no plan reply within", "plan failed (no reply in time)"),
    ("InfrastructureFailure", "no reset reply within", "reset failed (no reply in time)"),
    ("ProtocolError", "plan reply", "plan failed (malformed reply)"),
    ("ProtocolError", "reset reply", "reset failed (malformed reply)"),
    ("ProtocolError", "reset was not acknowledged", "reset failed (malformed reply)"),
    ("PolicyError", "policy returned an invalid action chunk", "plan failed (malformed reply)"),
    ("PolicyError", "inference failed:", "plan failed (client error)"))
# A camera that stalls during a plan reaches the runner wrapped by the executor: "PolicyError: inference failed: <the
# StaleImage message of run_pi_policy.image_ages / capture>".
STALE_IMAGE_MESSAGE = re.compile(r"inference failed: (?:no \S+ image yet|\S+ image is \S+s old)")


def console_reason(reason, outcome="infrastructure"):
    """What the blinded operator sees of an infrastructure failure: a fixed category followed by "; details in
    episodes.jsonl", never the reason's message (server text, connection names); a reason of another kind shows its
    exception class only. An attempt that ran to its end (outcome other than "infrastructure") or a missing reason
    is an infrastructure failure because of its refresh slots (classify)."""
    if not reason or outcome != "infrastructure":
        return "request failures above the limit" + CONSOLE_DETAILS
    name, separator, message = reason.partition(": ")
    if not separator or not name.isidentifier():
        return "infrastructure failure" + CONSOLE_DETAILS
    if name == "PolicyError" and STALE_IMAGE_MESSAGE.fullmatch(message.strip()):
        return CONSOLE_BY_CLASS["StaleImage"] + CONSOLE_DETAILS
    if name in CONSOLE_BY_CLASS:
        return CONSOLE_BY_CLASS[name] + CONSOLE_DETAILS
    for kind, prefix, category in CONSOLE_BY_MESSAGE:
        if name == kind and message.startswith(prefix):
            return category + CONSOLE_DETAILS
    return name + CONSOLE_DETAILS


# The log records the blinded operator may see. Everything goes to <log_dir>/client.log.
CLIENT_LOG = "client.log"
CLIENT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"


class ConsoleFilter(logging.Filter):
    """Console records: those of this runner, forceguide_client and the other aloha_scripts modules (the root logger
    and forceguide_client.DEBUG_LOG), unless marked FILE_ONLY (console_text None), and none while an attempt runs
    (trial_running, ClientLog.trial): a record's mere timing can tell the condition (a reconnect of the refresh
    connection, a follower step limit at a refresh splice), and the runner's own lines say what the operator needs.
    Library records stay in client.log: websockets also logs from the refresh connection's thread, a Python warning
    prints its source line, and ROS prints its own rosout records."""

    def __init__(self):
        super().__init__()
        self.trial_running = False

    def filter(self, record):
        if self.trial_running or getattr(record, "console_text", "") is None:
            return False
        return record.name in ("root", DEBUG_LOG.name) or record.name.startswith(DEBUG_LOG.name + ".")


class ConsoleFormatter(logging.Formatter):
    """'LEVEL message', with a record's neutral console_text in place of its message, and never a traceback or a
    stack (their frames and messages can name the condition's code path; client.log has them)."""

    def format(self, record):
        text = getattr(record, "console_text", None)
        return "{} {}".format(record.levelname, record.getMessage() if text is None else text)


class ClientLog:
    """The session's logging, in place of logging.basicConfig: every record (forceguide_client.DEBUG_LOG at DEBUG,
    the rest at INFO, Python warnings) appended to <log_dir>/client.log with timestamps, threads and tracebacks, and
    the operator-facing ones on the console (ConsoleFilter, ConsoleFormatter). attach() runs again after
    rospy.init_node, whose logging config (logging.config.fileConfig) replaces the root logger's handlers and level;
    close() restores what it changed."""

    def __init__(self, log_dir, stream=None):
        self.path = os.path.join(log_dir, CLIENT_LOG)
        # mode "a": sessions append, and after fileConfig closed the handler its next emit reopens the file
        self.file = logging.FileHandler(self.path, mode="a", encoding="utf-8")
        self.file.setLevel(logging.DEBUG)
        self.file.setFormatter(logging.Formatter(CLIENT_LOG_FORMAT))
        self.console = logging.StreamHandler(stream)  # None: sys.stderr, as logging.basicConfig
        self.console.setLevel(logging.INFO)
        self.console_filter = ConsoleFilter()
        self.console.addFilter(self.console_filter)
        self.console.setFormatter(ConsoleFormatter())
        self._levels = {logger: logger.level for logger in (logging.getLogger(), DEBUG_LOG)}
        self.trials_started = False  # main(): after the first attempt an escaping exception stays off the console
        logging.captureWarnings(True)
        self.attach()

    def attach(self):
        root = logging.getLogger()
        for handler in (self.file, self.console):
            if handler not in root.handlers:
                root.addHandler(handler)
        if root.getEffectiveLevel() > logging.INFO:
            root.setLevel(logging.INFO)
        DEBUG_LOG.setLevel(logging.DEBUG)

    @contextlib.contextmanager
    def trial(self):
        """While an attempt runs, no log record reaches the console (ConsoleFilter); client.log keeps them all."""
        self.trials_started = True
        self.console_filter.trial_running = True
        try:
            yield
        finally:
            self.console_filter.trial_running = False

    def close(self):
        root = logging.getLogger()
        for handler in (self.file, self.console):
            root.removeHandler(handler)
        self.file.close()
        for logger, level in self._levels.items():
            logger.setLevel(level)
        logging.captureWarnings(False)


def infra_frac(summary):
    """Fraction of the refresh slots lost to infrastructure faults (executor summary; recomputed from the counts for
    a summary written before "infra_frac" existed)."""
    if "infra_frac" in summary:
        return float(summary["infra_frac"] or 0.0)
    counts = summary.get("counts") or {}
    slots = summary.get("slots") or 0
    infra = sum(counts.get(status, 0) for status in INFRA_REFRESH_STATUSES) + counts.get("malformed", 0)
    return infra / slots if slots else 0.0


def classify(outcome, summary, max_bad_frac):
    """quit | infrastructure_failure | labelled (the operator decides success / failure / re-run).

    Only real infrastructure faults make an attempt an infrastructure failure (re-run): the outcome "infrastructure"
    (plan error, stale camera or tactile at a plan, changed server) or more than max_bad_frac of the refresh slots
    lost to failed / error / skipped-stale / skipped-disabled / malformed refreshes. Rejected, late, dropped and
    skipped-in-flight refreshes come from the method and its compute time (they grow with eta, K and contact):
    protocol_flags() records them, they never exclude or re-run a trial (robot review 3, exp 1)."""
    if outcome == "quit":
        return "quit"
    if outcome == "infrastructure":
        return "infrastructure_failure"
    if summary.get("slots") and infra_frac(summary) > max_bad_frac:
        return "infrastructure_failure"
    return "labelled"


def protocol_flags(summary, max_bad_frac):
    """Method / timing outcomes of an attempt's refreshes, kept as covariates (never a re-run cause): flag
    "bad_refresh_frac" when more than max_bad_frac of the slots were lost or late (the old exclusion rule, now the
    analysis' sensitivity analysis) and "late" when the late-reply fraction is above LATE_FLAG_FRAC."""
    counts = summary.get("counts") or {}
    flags = []
    if summary.get("slots") and summary.get("bad_frac", 0.0) > max_bad_frac:
        flags.append("bad_refresh_frac")
    if summary.get("late_flag"):
        flags.append("late")
    return {"flags": flags, "bad_frac": summary.get("bad_frac", 0.0), "protocol_frac": summary.get("protocol_frac"),
            "infra_frac": infra_frac(summary), "late_frac": summary.get("late_frac"),
            "counts": {status: counts.get(status, 0) for status in PROTOCOL_REFRESH_STATUSES + ("late", "late_any")},
            "max_bad_refresh_frac": max_bad_frac}


def answered(record):
    """A refresh round trip the server answered with a result: no transport failure and no error reply (an error
    reply returns fast and would bias the RTT statistics and the lead low)."""
    return record.get("transport") == "ok" and record.get("status") not in ("error", "failed")


def read_attempts(path, schedule_sha256=None):
    """{trial_id: [rows]} of an existing episodes.jsonl (resume): the attempts of this schedule (trials are keyed
    by (schedule_sha256, trial_id); rows of older clients without schedule_sha256 count as this schedule's)."""
    attempts = {}
    if not os.path.exists(path):
        return attempts
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if schedule_sha256 and row.get("schedule_sha256") not in (None, schedule_sha256):
                    continue
                attempts.setdefault(str(row.get("trial_id")), []).append(row)
    return attempts


def refresh_intervals(refreshes):
    """Policy steps between consecutive plan updates inside each chunk (the chunk-start plan counts as k = 0), over
    the APPLIED refreshes: the effective refresh interval the robot actually got."""
    by_chunk = {}
    for record in refreshes:
        if record.get("status") == "applied" and record.get("s") is not None:
            by_chunk.setdefault(record.get("chunk"), []).append(int(record["s"]))
    intervals = []
    for starts in by_chunk.values():
        points = [0] + sorted(starts)
        intervals.extend(b - a for a, b in zip(points[:-1], points[1:]))
    return intervals


def trial_stats(ctx):
    plans = [r for r in ctx.requests if r.get("type") == "plan"]
    refreshes = [r for r in ctx.requests if r.get("type") == "refresh"]
    intervals = refresh_intervals(refreshes)
    return {"plan_rtt_ms": summarize_ms(r.get("rtt_ms") for r in plans),
            "plan_pause_ms": summarize_ms(r.get("pause_ms") for r in plans),
            "pause_overruns": sum(1 for r in plans if r.get("pause_overrun")),
            "refresh_rtt_ms": summarize_ms(r.get("rtt_tick_ms") for r in refreshes if answered(r)),
            "refresh_K": dict(_count(r.get("K") for r in refreshes if answered(r))),
            "refresh_errors": sum(1 for r in refreshes if r.get("status") in ("error", "failed")),
            "refresh_interval_steps": (float(np.mean(intervals)) if intervals else None),
            "splice_late_steps": summarize_ms(r.get("late") for r in refreshes if r.get("status") == "applied"),
            "overruns": sum(1 for t in ctx.ticks if t.get("overrun")),
            "ticks": len(ctx.ticks)}


def _count(values):
    counts = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts.items()


# --------------------------------------------------------------------------------------------
# Trial set-up shared by warm-up, preflight and trials
# --------------------------------------------------------------------------------------------

class Session:
    """Everything that lives for the whole run (connections, campaign settings, adapter, tactile rig)."""

    def __init__(self, args, client, campaign, adapter, rig, guard, control_dt, action_horizon, capture_frame,
                 camera_ages=None, clock=time.monotonic, sleep=time.sleep):
        self.args = args
        self.client = client
        self.info = client.info
        self.campaign = campaign
        self.adapter = adapter
        self.rig = rig
        self.guard = guard
        self.control_dt = control_dt
        self.action_horizon = action_horizon
        self.capture_frame = capture_frame
        self.camera_ages = camera_ages
        self.clock = clock
        self.sleep = sleep
        self.advance = control_dt * campaign.policy_hz
        self.refresh_period_s = (campaign.refresh_every / campaign.policy_hz) if campaign.refresh_every else None
        # Written into every episodes.jsonl row: schedule_sha256, plan_pause_source, preflight_check (set by main)
        self.run_record = {}
        self.client_log = None  # main's ClientLog: the trial loop keeps log records off the console during attempts

    def trial(self, condition, noise_id, measured_fn, plan_timeout, labels):
        settings = self.campaign.settings(condition)
        self.rig.reset(settings.spec, allow_legacy_pad=self.campaign.allow_legacy_pad)
        payload = settings.make_payload(self.rig.history)
        noise = NoiseSchedule(self.args.seed, noise_id, self.adapter.action_horizon, self.adapter.model_action_dim)
        return TrialContext(self.client, self.adapter, condition, payload, monitor=self.rig.monitor, noise=noise,
                            measured_fn=measured_fn, plan_timeout=plan_timeout,
                            max_tactile_age=self.args.max_tactile_age, labels=labels)

    def executor(self, ctx, refresh_timeout_s, request_timeout_s=None, lead=None):
        condition = self.info.condition(ctx.condition)
        refresh = condition.mode == "refresh"
        lead = lead if lead is not None else self.campaign.lead
        return RefreshingChunkExecutor(
            ctx.plan, self.campaign.policy_hz, self.control_dt, action_horizon=self.action_horizon,
            clock=self.clock, mode=condition.mode, refresh_every=self.campaign.refresh_every if refresh else None,
            lead=lead if refresh else None, transport=self.client.worker if refresh else None,
            decode=self.adapter.decode, guard=self.guard, can_refresh=ctx.can_refresh,
            refresh_timeout_s=refresh_timeout_s, log_fn=ctx.log_request, sidecar=ctx.sidecar, payload=ctx.payload,
            connect_budget_s=self.refresh_period_s, request_timeout_s=request_timeout_s)

    def dry_run(self, condition, chunks, timeout, phase, measured_fn, lead=None):
        """Robot static: reset + `chunks` chunks of the real loop (plans and refreshes), commands not sent."""
        ctx = self.trial(condition, 0, measured_fn, timeout, {"phase": phase, "condition": condition})
        ctx.log_request(self.client.reset(condition, self.campaign.policy_hz, timeout))
        executor = self.executor(ctx, refresh_timeout_s=None, request_timeout_s=timeout, lead=lead)
        length = min(self.action_horizon, self.info.action_horizon)
        steps = chunks * ticks_per_chunk(length, self.advance)
        outcome, reason = run_forceguide_episode(executor, DryFollower(measured_fn), NullKeyboard(),
                                                 self.capture_frame, self.rig, steps, self.control_dt,
                                                 plan_pause_s=0.0, clock=self.clock, sleep=self.sleep, ctx=ctx)
        executor.finish(wait_s=min(timeout, 5.0), sleep=self.sleep)  # the last reply is logged, not leaked
        return ctx, executor, outcome, reason


def warm_up(session, measured_fn, log_path, codecs=None):
    """reset -> plan -> refreshes -> observe for every condition with the warm-up timeout (the server compiled
    everything at startup; this checks the real round trips before round 1; every trial sends its own reset after
    this). Raises on failure; the console names no condition (the operator is blinded)."""
    records = []
    start = time.monotonic()
    for index, condition in enumerate(session.campaign.names):
        t0 = time.monotonic()
        ctx, executor, outcome, reason = session.dry_run(condition, 1, session.args.warmup_timeout, "warmup",
                                                         measured_fn, lead=session.campaign.default_lead)
        if outcome != "done":
            write_jsonl(log_path, records + ctx.requests)
            raise InfrastructureFailure("warm-up of condition #{} failed: {} (details in {})".format(
                index + 1, reason, log_path))
        counts = executor.summary().get("counts") or {}
        faults = {status: counts.get(status, 0) for status in ("error", "failed", "malformed") if counts.get(status)}
        if faults:  # e.g. every refresh refused for a lead the server's calibration does not allow (robot review 1)
            write_jsonl(log_path, records + ctx.requests)
            first = next((str(r.get("reason")) for r in ctx.requests if r.get("type") == "refresh"
                          and (r.get("status") in ("error", "failed") or r.get("malformed"))), "")
            raise InfrastructureFailure("warm-up of condition #{}: refresh requests failed {} (first: {}; details in "
                                        "{})".format(index + 1, faults, (first.splitlines() or [""])[0][:200],
                                                     log_path))
        observe = None
        if ctx.payload.keys and session.client.worker is not None:
            step = session.rig.history.next_step - 1
            stale = ctx.stale_reasons(step)  # observe follows the refresh rule: never upload stale tactile
            if stale:
                observe = {"status": "skipped-stale", "error": "; ".join(stale), "reply": None}
            else:
                result = session.client.observe(ctx.payload, step, session.args.warmup_timeout)
                observe = {"status": result.status, "error": result.error,
                           "reply": (result.reply or {}).get("forceguide"), "client": result.timing}
        records.extend(ctx.requests)
        records.append({"type": "warmup", "condition": condition, "seconds": time.monotonic() - t0,
                        "refresh": executor.summary(), "observe": observe, "tactile_codecs": codecs})
    write_jsonl(log_path, records)
    print("Warm-up done in {:.1f} s.".format(time.monotonic() - start))


def run_preflight(session, chunks, measured_fn):
    """Latency preflight: plan and refresh RTT per condition (and per K), the lead the refresh conditions need, and
    the plan pause (p99 plan RTT of the slowest condition).

    The lead is computed from the larger of the overall p95 refresh RTT and the p95 of the slowest K level seen (the
    server picks K after the request is sent, so the lead must cover K_max). The client cannot force K levels: for
    dynamic / random K conditions only the levels the static robot produced are measured (listed in the report);
    measure K_max separately with the server's bench if it did not occur.

    Only answered refreshes enter the RTT statistics: error replies (a refused lead, a server exception) and failed
    requests are listed as problems instead (they return fast and would understate the lead)."""
    campaign = session.campaign
    lead = campaign.default_lead
    per_condition = {}
    refresh_rtts = []
    by_k_all = {}
    plan_p99 = []
    problems = []
    for condition in campaign.names:
        ctx, executor, outcome, reason = session.dry_run(condition, chunks, session.args.infer_timeout, "preflight",
                                                         measured_fn, lead=lead)
        plans = [r for r in ctx.requests if r.get("type") == "plan"]
        refreshes = [r for r in ctx.requests if r.get("type") == "refresh" and answered(r)]
        faults = [r for r in ctx.requests if r.get("type") == "refresh"
                  and (r.get("status") in ("error", "failed") or r.get("malformed"))]
        if faults:
            problems.append("condition {}: {} of its refresh requests failed or got an error reply (first: {})".format(
                condition, len(faults), (str(faults[0].get("reason")).splitlines() or [""])[0][:200]))
        by_k = {}
        for record in refreshes:
            by_k.setdefault(str(record.get("K")), []).append(record.get("rtt_tick_ms"))
            by_k_all.setdefault(str(record.get("K")), []).append(record.get("rtt_tick_ms"))
        entry = {"mode": session.info.condition(condition).mode, "outcome": outcome, "reason": reason,
                 "plan_rtt_ms": summarize_ms(r.get("rtt_ms") for r in plans),
                 "plan_server_ms": summarize_ms((r.get("server_timing") or {}).get("infer_ms") for r in plans),
                 "refresh_rtt_ms": summarize_ms(r.get("rtt_tick_ms") for r in refreshes),
                 "refresh_rtt_ms_by_K": {k: summarize_ms(v) for k, v in by_k.items()},
                 "refresh_errors": len(faults), "refresh": executor.summary()}
        per_condition[condition] = entry
        if outcome != "done":
            problems.append("condition {}: {}".format(condition, reason))
        if entry["plan_rtt_ms"].get("n"):
            plan_p99.append(entry["plan_rtt_ms"]["p99"])
        refresh_rtts.extend(r.get("rtt_tick_ms") for r in refreshes)
    refresh = summarize_ms(refresh_rtts)
    by_k_summary = {k: summarize_ms(v) for k, v in by_k_all.items()}
    report = {"conditions": per_condition, "advance": session.advance, "control_dt": session.control_dt,
              "policy_hz": campaign.policy_hz, "refresh_every": campaign.refresh_every,
              "configured_lead": campaign.lead, "measured_lead": lead if campaign.refresh_conditions else None,
              "calibrated_leads": dict(campaign.calibrated_leads), "min_lead": campaign.min_lead,
              "refresh_rtt_ms": refresh, "refresh_rtt_ms_by_K": by_k_summary,
              "K_levels_seen": sorted(k for k in by_k_all if k not in ("None", "0")),
              "plan_rtt_p99_ms": max(plan_p99) if plan_p99 else None}
    if plan_p99:
        report["plan_pause_s_recommended"] = math.ceil(max(plan_p99) * 1.1 / 10.0) / 100.0
    if campaign.refresh_conditions:
        if not refresh.get("n"):
            problems.append("no refresh round trip succeeded")
        else:
            rtt_ms = max([refresh["p95"]] + [s["p95"] for k, s in by_k_summary.items()
                                             if s.get("n") and k not in ("None", "0")])
            length = min(session.action_horizon, session.info.action_horizon)
            needed = required_lead(session.advance, rtt_ms / 1000.0, session.control_dt, horizon=length,
                                   refresh_every=campaign.refresh_every)
            report["lead_rtt_ms"] = rtt_ms
            report["required_lead"] = needed
            report["refresh_period_ms"] = 1000.0 * session.refresh_period_s
            if needed is None:
                problems.append("no lead works at a p95 refresh RTT of {:.0f} ms".format(rtt_ms))
            elif campaign.lead is not None and campaign.lead < needed:
                problems.append("--lead {} is too small: the p95 refresh RTT {:.0f} ms needs lead {}".format(
                    campaign.lead, rtt_ms, needed))
            elif campaign.lead is None:
                report["note"] = ("no --lead given (measured with lead {}); choose a lead >= {}, calibrate eta for it "
                                  "(offline_guidance_check --lead L), serve the calibration, run the campaign with "
                                  "--lead L".format(lead, max(needed, campaign.min_lead or 1)))
            if rtt_ms >= 1000.0 * session.refresh_period_s:
                problems.append("p95 refresh RTT {:.0f} ms is not below the refresh period {:.0f} ms".format(
                    rtt_ms, 1000.0 * session.refresh_period_s))
    report["problems"] = problems
    report["ok"] = not problems
    return report


def _report_written(report, path):
    """When a preflight report was written (its "written" field, else the file's mtime)."""
    try:
        return datetime.datetime.strptime(str(report["written"])[:19], "%Y-%m-%dT%H:%M:%S")
    except (KeyError, TypeError, ValueError):
        return datetime.datetime.fromtimestamp(os.path.getmtime(path))


def check_preflight_report(path, campaign, advance, force, required=False, max_age_h=24.0, plan_pause_s=None,
                           now=None):
    """Trial mode: the timing preflight report (preflight.json) the campaign relies on. Returns (report, check).

    Always refuses (unless force) a --lead smaller than the report's required lead (same rate and cadence). For a
    campaign run (required: a --schedule run, exp 10) the report is mandatory: it must exist, be at most max_age_h
    old, have passed, cover every condition of the campaign at the same rate and cadence, and its p99 plan RTT must
    fit in the chunk-start pause (plan_pause_s; None = the report's recommendation, which does) - else SystemExit
    unless force. check (path, sha256, written, age, problems, forced) goes to sessions.jsonl and every episode."""
    check = {"path": path, "required": bool(required), "exists": False, "sha256": None, "written": None,
             "age_h": None, "problems": [], "forced": False}
    problems = check["problems"]
    report = None
    if not path or not os.path.exists(path):
        problems.append("no preflight report ({}): run --preflight first".format(path))
    else:
        with open(path) as handle:
            report = json.load(handle)
        written = _report_written(report, path)
        age_h = ((now or datetime.datetime.now()) - written).total_seconds() / 3600.0
        check.update(exists=True, sha256=file_sha256(path), written=written.isoformat(), age_h=age_h,
                     ok=report.get("ok"), required_lead=report.get("required_lead"),
                     plan_rtt_p99_ms=report.get("plan_rtt_p99_ms"),
                     plan_pause_s_recommended=report.get("plan_pause_s_recommended"))
        same = (abs(float(report.get("advance") or 0) - advance) < 1e-9
                and report.get("refresh_every") == campaign.refresh_every)
        needed = report.get("required_lead")
        if campaign.refresh_conditions and needed is not None and same and campaign.lead < needed:
            message = "--lead {} is smaller than the preflight's required lead {} ({})".format(campaign.lead, needed,
                                                                                               path)
            if not force:
                raise SystemExit(message)
            logging.warning("%s - continuing because of --force", message)
            check["forced"] = True
        if age_h > max_age_h:
            problems.append("the preflight report is {:.1f} h old (> --preflight_max_age_h {})".format(age_h,
                                                                                                    max_age_h))
        if not report.get("ok"):
            problems.append("the preflight was refused: {}".format(report.get("problems")))
        missing = [name for name in campaign.names if name not in (report.get("conditions") or {})]
        if missing:
            problems.append("the preflight did not measure conditions {}".format(missing))
        if not same:
            problems.append("the preflight ran at advance {} / refresh_every {}, the campaign at {} / {}".format(
                report.get("advance"), report.get("refresh_every"), advance, campaign.refresh_every))
        p99 = report.get("plan_rtt_p99_ms")
        if plan_pause_s is not None and p99 and 1000.0 * plan_pause_s < p99:
            problems.append("--plan_pause_s {} is shorter than the preflight's p99 plan RTT {:.0f} ms: the chunk-start "
                            "pause would differ between conditions".format(plan_pause_s, p99))
    if problems:
        message = "preflight report: " + "; ".join(problems)
        if required and not force:
            raise SystemExit(message + " (a --schedule campaign needs a recent, passed preflight; --force overrides)")
        logging.warning("%s%s", message, " - continuing because of --force" if required else "")
        check["forced"] = check["forced"] or bool(required)
    return report, check


def check_resume(episodes_path, schedule_sha256, plan_pause_s, lead, force, run_label=None, seed=None):
    """A resumed campaign (episodes.jsonl exists) keeps its schedule, its chunk-start pause, its lead and its noise
    seed: an eta is calibrated for one lead, the pause must be the same in every trial, and the seed draws the plan
    noise that pairs a trial across conditions (default_rng([seed, noise_id, plan_index])). A run without a schedule
    file also keeps its run_label (part of its schedule_sha256): another label is refused like another schedule.
    SystemExit unless force; returns what was compared (sessions.jsonl)."""
    rows = []
    if os.path.exists(episodes_path):
        with open(episodes_path) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    record = {"previous_attempts": len(rows), "problems": [], "forced": False}
    if not rows:
        return record
    shas = sorted({str(r["schedule_sha256"]) for r in rows if r.get("schedule_sha256")})
    pauses = sorted({float(r["plan_pause_s"]) for r in rows if r.get("plan_pause_s") is not None})
    leads = sorted({int(r["lead"]) for r in rows if r.get("lead") is not None})
    labels = sorted({str(r["run_label"]) for r in rows if r.get("run_label")})
    seeds = sorted({int(r["seed"]) for r in rows if r.get("seed") is not None})
    record.update(schedule_sha256=shas, plan_pause_s=pauses, lead=leads, run_label=labels, seed=seeds,
                  rows_without_schedule_sha256=sum(1 for r in rows if not r.get("schedule_sha256")))
    if record["rows_without_schedule_sha256"]:
        logging.warning("%s: %d attempts were logged without schedule_sha256 (older client); their schedule cannot "
                        "be verified", episodes_path, record["rows_without_schedule_sha256"])
    if run_label is not None and any(label != run_label for label in labels):
        record["problems"].append(
            "this log dir holds attempts of run label {}, now {!r} (the run label is part of the trial identity: "
            "pass the run's --run_label to resume it)".format(labels, run_label))
    if any(sha != schedule_sha256 for sha in shas):
        record["problems"].append("this log dir holds attempts of another schedule (sha256 {}{})".format(
            [sha for sha in shas if sha != schedule_sha256],
            "; logged without a run label: a --schedule campaign, or a --condition run of a client older than run "
            "labels, which this client cannot resume" if run_label is not None and not labels else ""))
    if any(abs(pause - plan_pause_s) > 1e-9 for pause in pauses):
        record["problems"].append("the campaign ran with --plan_pause_s {}, now {} (pass the campaign's pause "
                                  "explicitly)".format(pauses, plan_pause_s))
    if lead is not None and any(value != lead for value in leads):
        record["problems"].append("the campaign ran with --lead {}, now {}".format(leads, lead))
    if seed is not None and any(value != int(seed) for value in seeds):
        record["problems"].append("the campaign ran with --seed {}, now {} (the seed draws the plan noise that pairs "
                                  "the trials across conditions: pass the campaign's --seed)".format(seeds, seed))
    if record["problems"]:
        message = "resuming {}: {}".format(episodes_path, "; ".join(record["problems"]))
        if not force:
            raise SystemExit(message + " (use another --log_dir for a new campaign; --force overrides)")
        logging.warning("%s - continuing because of --force", message)
        record["forced"] = True
    return record


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

def make_schedule_main(args, parser):
    if not args.schedule_conditions or not args.schedule_inits:
        parser.error("--make_schedule needs --schedule_conditions and --schedule_inits")
    rows = make_schedule(args.schedule_conditions, parse_inits(args.schedule_inits), seed=args.schedule_seed,
                         repeats=args.schedule_repeats)
    write_schedule(args.make_schedule, rows)
    print("Wrote {} trials ({} conditions x {} initial conditions x {} repeats) to {}".format(
        len(rows), len(set(args.schedule_conditions)), len(parse_inits(args.schedule_inits)), args.schedule_repeats,
        args.make_schedule))
    return 0


def wait_for_tactile(rig, timeout_s, force, sleep=time.sleep):
    """All campaign sensors publishing, with the raw codec (the training frames were never JPEG-coded)."""
    deadline = time.monotonic() + timeout_s
    snapshot = rig.snapshot()
    while any(frame is None for frame in snapshot.values()):
        if time.monotonic() > deadline:
            missing = [sensor for sensor, frame in snapshot.items() if frame is None]
            raise RuntimeError("no tactile frames from {} within {:.0f} s - is the XENSE publisher running?".format(
                missing, timeout_s))
        sleep(0.05)
        snapshot = rig.snapshot()
    codecs = {sensor: (frame.meta or {}).get("image_codec") for sensor, frame in snapshot.items()}
    bad = {sensor: codec for sensor, codec in codecs.items() if codec not in (None, "raw")}
    if bad:
        message = "tactile publisher codec {} (needs image_codec:=raw)".format(bad)
        if not force:
            raise RuntimeError(message)
        logging.warning("%s - continuing because of --force", message)
    return codecs


def main(argv=None, input_fn=None):
    """Returns the exit code. The session's client.log (ClientLog) is closed on the way out, after it recorded the
    traceback of an exception that ends the run. Ctrl-C returns 130, and any other exception once an attempt has
    started returns 1, with one neutral console line instead of Python's traceback: its frames and message could show
    the code path the trial was in (e.g. refresh-only bookkeeping in executor.finish or write_trial). Before the
    first attempt (setup refusals) the exception is raised again, so its message stays on the console."""
    logs = []
    try:
        return _main(argv, input_fn, logs)
    except KeyboardInterrupt:
        if not logs:
            raise
        logging.error("interrupted (Ctrl-C)", exc_info=True, extra=FILE_ONLY)
        print("Interrupted (Ctrl-C); the traceback is in {}.".format(logs[-1].path))
        return 130
    except Exception:
        if logs:
            logging.error("the run stopped on an exception", exc_info=True, extra=FILE_ONLY)
            if logs[-1].trials_started:
                print("Stopped by an unexpected client error; the traceback is in {}.".format(logs[-1].path))
                return 1
        raise
    finally:
        for log in logs:
            log.close()


def _main(argv, input_fn, logs):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.make_schedule:
        return make_schedule_main(args, parser)
    if not args.host:
        parser.error("--host is required")
    if not args.schedule and not args.condition:
        parser.error("pass --schedule trials.jsonl or --condition NAME")
    if args.schedule and not args.log_dir:
        # resume, the preflight report and the trial identity (schedule_sha256, trial_id, attempt) live there
        parser.error("a --schedule run needs an explicit --log_dir (the campaign's log directory)")
    if args.schedule and args.run_label is not None:
        parser.error("--run_label labels a run without a schedule file (--condition); a --schedule campaign's trials "
                     "are identified by its schedule file (schedule_sha256)")
    # The log dir is decided before the trial rows: a run without a schedule file is labelled by it (contract C9)
    log_dir = args.log_dir or os.path.join("forceguide_logs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_label = None
    if not args.schedule:
        try:
            run_label = run_label_for(log_dir, args.run_label)
        except ValueError as exc:
            parser.error(str(exc))
    os.makedirs(log_dir, exist_ok=True)
    client_log = ClientLog(log_dir)  # before the first connection: a server that is not up is logged there too
    logs.append(client_log)
    print("Full log (not for the blinded operator): {}".format(client_log.path))
    logging.info("run_forceguide_policy %s", " ".join(sys.argv[1:] if argv is None else argv), extra=FILE_ONLY)
    input_fn = input_fn or input

    # Hardware imports here, so --help, --make_schedule and the tests work without ROS.
    import rospy
    import tactile_stream
    from constants import DT, PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, START_ARM_POSE
    from hil_devices import FollowerCommander
    from one_side_teleop import KeyboardControl
    from real_env import make_real_env
    from robot_utils import move_arms, move_grippers
    from sleep_plus import shut_down_puppet_robots

    main_conn = ForceGuideConnection(args.host, args.port, connect_timeout_s=args.connect_timeout,
                                     infer_timeout_s=args.infer_timeout, name="main")
    try:  # everything that can refuse the run, before the refresh connection and its worker exist
        info = ServerInfo.from_metadata(main_conn.metadata)
        if args.schedule:
            rows = load_schedule(args.schedule, info)
            schedule_sha256 = file_sha256(args.schedule)
        else:
            info.condition(args.condition)
            rows = generated_rows(args.condition, args.num_episodes, run_label)
            schedule_sha256 = rows_sha256(rows)
        campaign_run = bool(args.schedule) and not args.preflight  # exp 10: pause and preflight are mandatory
        campaign = CampaignConfig(info, [row["condition"] for row in rows], lead=args.lead,
                                  policy_hz=args.policy_hz, refresh_every=args.refresh_every, force=args.force,
                                  sensor_overrides=parse_sensor_overrides(args.tactile_sensor),
                                  require_lead=not args.preflight)
        os.makedirs(os.path.join(log_dir, "trials"), exist_ok=True)
        reset_pose = list(main_conn.metadata.get("reset_pose") or START_ARM_POSE)[:6]
        adapter = ForceGuideAdapter.from_info(info, prompt=args.prompt, hold_pose=reset_pose)
        action_horizon = args.action_horizon or info.action_horizon
        if action_horizon > info.action_horizon:
            parser.error("--action_horizon {} exceeds the server's horizon {}".format(action_horizon,
                                                                                     info.action_horizon))
        calibration_source = None
        if adapter.state_gripper == "vtla":
            expected = gripper_calibration_from_metadata(info.raw)
            calibration_source = "server metadata"
            if expected is None and args.dataset_metadata:
                with open(args.dataset_metadata) as handle:
                    expected = gripper_calibration_from_metadata(json.load(handle))
                calibration_source = args.dataset_metadata
            if expected is None:
                calibration_source = "board_insertion values built into forceguide_client"
            check_gripper_calibration(expected)
        guard = SpliceGuard(adapter.active_joints, adapter.active_grippers, args.max_splice_jump,
                            args.max_plan_step, args.max_gripper_jump)
        advance = DT * campaign.policy_hz
        report_path = args.preflight_report or os.path.join(log_dir, "preflight.json")
        preflight, preflight_check = None, None
        if not args.preflight:
            preflight, preflight_check = check_preflight_report(
                report_path, campaign, advance, args.force, required=campaign_run,
                max_age_h=args.preflight_max_age_h, plan_pause_s=args.plan_pause_s)
        plan_pause_s, pause_source = args.plan_pause_s, "--plan_pause_s"
        if plan_pause_s is None:
            recommended = (preflight or {}).get("plan_pause_s_recommended")
            plan_pause_s, pause_source = float(recommended or 0.0), ("preflight" if recommended else "none")
        if not args.preflight and plan_pause_s <= 0.0:
            message = ("chunk-start pauses are not padded (--plan_pause_s {} and no preflight recommendation), so "
                       "they differ between conditions".format(args.plan_pause_s))
            if campaign_run and not args.force:
                raise SystemExit(message + ": pass --plan_pause_s or run --preflight first (--force overrides)")
            logging.warning(message)
        resume_check = None
        if not args.preflight:
            resume_check = check_resume(os.path.join(log_dir, "episodes.jsonl"), schedule_sha256, plan_pause_s,
                                        campaign.lead, args.force, run_label=run_label, seed=args.seed)
    except BaseException:
        main_conn.close()
        raise
    refresh_budget = (campaign.refresh_every / campaign.policy_hz) if campaign.refresh_every else 0.2
    try:
        refresh_conn = ForceGuideConnection(args.host, args.port, connect_timeout_s=args.connect_timeout,
                                            infer_timeout_s=args.infer_timeout, close_timeout_s=0.2, name="refresh")
        client = ForceGuideClient(main_conn, refresh_conn, info, hard_timeout_s=args.infer_timeout,
                                  refresh_budget_s=refresh_budget)
    except BaseException:
        main_conn.close()
        raise
    session_record = {"session": client.session, "started": datetime.datetime.now().isoformat(),
                      "argv": list(sys.argv if argv is None else argv), "host": args.host, "port": args.port,
                      "client_git_rev": git_rev(), "metadata": main_conn.metadata, "campaign": campaign.to_dict(),
                      "schedule": args.schedule, "schedule_sha256": schedule_sha256, "run_label": run_label,
                      "seed": args.seed, "plan_pause_s": plan_pause_s, "plan_pause_source": pause_source,
                      "preflight_check": preflight_check, "resume_check": resume_check,
                      "refresh_timeout_s": args.refresh_timeout,
                      "action_horizon": action_horizon, "control_dt": DT, "advance": advance,
                      "gripper_calibration_source": calibration_source,
                      "thresholds": {"max_splice_jump": args.max_splice_jump, "max_plan_step": args.max_plan_step,
                                     "max_gripper_jump": args.max_gripper_jump,
                                     "max_tactile_age": args.max_tactile_age,
                                     "max_bad_refresh_frac": args.max_bad_refresh_frac}}
    write_jsonl(os.path.join(log_dir, "sessions.jsonl"), [session_record], mode="a")
    run_note = ", run label {}".format(run_label) if run_label is not None else ""
    logging.info("Session %s: conditions %s, lead %s, refresh_every %s, plan pause %.3f s%s", client.session,
                 campaign.names, campaign.lead, campaign.refresh_every, plan_pause_s, run_note,
                 extra=console_text("Session {}: {} condition(s), plan pause {:.3f} s{}".format(
                     client.session, len(campaign.names), plan_pause_s, run_note)))

    try:  # robot and sensors (as run_pi_policy: a failure here closes the client, the arms were not moved)
        env = make_real_env(init_node=True, setup_robots=True)
        puppets = (env.puppet_bot_left, env.puppet_bot_right)
        streams = {sensor: tactile_stream.TactileStream.acquire(sensor, prefetch=(("rectify", "rgb"),))
                   for sensor in campaign.sensors}
    except BaseException:
        client.close()
        raise
    client_log.attach()  # rospy.init_node (make_real_env) replaced the root logger's handlers with ROS's log file
    rig = TactileRig(streams, campaign.history_capacity, args.max_tactile_age)
    clip_gripper = adapter.state_gripper == "vtla"
    observer = FollowerCommander(puppets, max_joint_step=None)  # measured() only, never sends

    def capture_frame():
        return capture(env, adapter.cameras, args.max_stream_age, rospy.get_time)

    def camera_ages():
        try:
            return image_ages(env.image_recorder, adapter.cameras, rospy.get_time())
        except StaleImage:
            return None

    def home():
        home_followers(puppets, reset_pose, move_arms, move_grippers,
                       (PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN), DT)

    def calibrate():
        if not campaign.sensors or args.no_tactile_calibration:
            return None
        print("Zeroing the tactile sensors - nothing may touch them...")
        blob = tactile_stream.calibrate_sensors(list(campaign.sensors))
        return None if not blob else {key: blob.get(key) for key in ("serials", "calibrated", "calibrated_at",
                                                                     "exported_at", "sdk_version")}

    session = Session(args, client, campaign, adapter, rig, guard, DT, action_horizon, capture_frame, camera_ages)
    session.run_record = {"schedule_sha256": schedule_sha256, "run_label": run_label,
                          "plan_pause_source": pause_source, "preflight_check": preflight_check}
    session.client_log = client_log
    keyboard = [None]
    final_lines = []  # printed after the followers are asleep, so the robot's shutdown output cannot bury them
    try:
        keyboard[0] = KeyboardControl()
        codecs = wait_for_tactile(rig, args.tactile_wait, args.force) if campaign.sensors else {}
        print("Homing the followers...")
        home()
        calibrate()
        if not args.no_warmup:
            print("Warming up every condition (robot static)...")
            warm_up(session, observer.measured, os.path.join(log_dir, "warmup.jsonl"), codecs)
        if args.preflight:
            print("Preflight: {} chunks per condition, robot static...".format(args.preflight))
            report = run_preflight(session, args.preflight, observer.measured)
            report.update(session=client.session, codecs=codecs, written=datetime.datetime.now().isoformat())
            with open(report_path, "w") as handle:
                json.dump(to_jsonable(report), handle, indent=2)
            final_lines.append(json.dumps(to_jsonable({key: report.get(key) for key in (
                "refresh_rtt_ms", "plan_rtt_p99_ms", "required_lead", "configured_lead", "plan_pause_s_recommended",
                "problems", "note")}), indent=2))
            final_lines.append("Preflight report: {} ({})".format(report_path, "OK" if report["ok"] else "REFUSED"))
            return 0 if report["ok"] or args.force else 2
        return run_trials(session, rows, args, log_dir, keyboard, KeyboardControl, input_fn, home, calibrate,
                          lambda: FollowerCommander(puppets, max_joint_step=args.max_joint_step or None,
                                                    clip_gripper=clip_gripper),
                          lambda planned, sent: clamp_flags(planned, sent, puppets, clip_gripper),
                          plan_pause_s, preflight, codecs=codecs)
    finally:
        client.close()
        if keyboard[0] is not None:
            keyboard[0].close()
        rig.close()
        print("Opening the grippers and moving the followers to the sleep pose...")
        try:
            shut_down_puppet_robots()
            print("Followers are at the sleep pose with torque off.")
        finally:
            for line in final_lines:
                print(line)


def run_trials(session, rows, args, log_dir, keyboard, keyboard_factory, input_fn, home, calibrate, make_follower,
               clamp_fn, plan_pause_s, preflight, codecs=None):
    """The trial loop: resume, attempts, operator outcome, logs. Returns the exit code.

    The console shows only trial ids, initial conditions and outcome categories, never the condition (blinding);
    the condition, its fingerprint and every refresh outcome go to episodes.jsonl."""
    episodes_path = os.path.join(log_dir, "episodes.jsonl")
    attempts = read_attempts(episodes_path, session.run_record.get("schedule_sha256"))
    client, campaign = session.client, session.campaign
    totals = {"slots": 0, "late_any": 0}
    try:
        return _run_trial_rows(session, rows, args, log_dir, keyboard, keyboard_factory, input_fn, home, calibrate,
                               make_follower, clamp_fn, plan_pause_s, preflight, codecs, episodes_path, attempts,
                               client, campaign, totals)
    finally:
        if totals["slots"]:
            late = totals["late_any"] / totals["slots"]
            if late > LATE_FLAG_FRAC:  # neutral words on the console: only refresh conditions have these replies
                logging.warning("%d of this session's %d refresh slots got a late reply (%.1f%% > %.0f%%)",
                                totals["late_any"], totals["slots"], 100.0 * late, 100.0 * LATE_FLAG_FRAC,
                                extra=FILE_ONLY)
                print("WARNING: {:.0%} of this session's deadline-bound server replies arrived late (> {:.0%}): "
                      "raise --lead (re-run --preflight) before using these trials.".format(late, LATE_FLAG_FRAC))


def _run_trial_rows(session, rows, args, log_dir, keyboard, keyboard_factory, input_fn, home, calibrate,
                    make_follower, clamp_fn, plan_pause_s, preflight, codecs, episodes_path, attempts, client,
                    campaign, totals):
    no_outcome = getattr(args, "no_outcome", False)
    for row in rows:
        previous = attempts.get(row["trial_id"], [])
        if any(r.get("status") in FINAL_STATUSES or (no_outcome and r.get("status") == "unlabelled")
               for r in previous):
            continue
        attempt = len(previous)
        while True:
            attempt += 1
            if attempt > args.max_attempts:
                print("Trial {}: {} attempts used up, moving on.".format(row["trial_id"], args.max_attempts))
                break
            print("=" * 80)
            print("Re-homing the followers...")
            home()
            tactile_calibration = calibrate()
            shown = " [{}]".format(row["condition"]) if getattr(args, "unblinded", False) else ""
            print("Trial {}{} (attempt {}): set initial condition {}. RIGHT = start, ESC = quit.".format(
                row["trial_id"], shown, attempt, row["init_condition_id"]))
            if keyboard[0].enabled and not wait_for_start(keyboard[0]):
                return 0
            follower = make_follower()
            labels = {"trial_id": row["trial_id"], "attempt": attempt}
            outcome, reason = "infrastructure", None
            executor = None
            # blinding: from the condition's setup to the end of the attempt, log records go to client.log only
            blinded_log = session.client_log is not None and not getattr(args, "unblinded", False)
            with session.client_log.trial() if blinded_log else contextlib.nullcontext():
                ctx = session.trial(row["condition"], row["noise_id"], follower.measured, args.infer_timeout, labels)
                started = datetime.datetime.now().isoformat()
                t0 = time.monotonic()
                try:
                    ctx.log_request(client.reset(row["condition"], campaign.policy_hz, args.infer_timeout,
                                                 trial=row["trial_id"], attempt=attempt))
                    executor = session.executor(ctx, refresh_timeout_s=args.refresh_timeout)
                    print("Running. RIGHT = end the trial, LEFT or r = abort, ESC/q = stop.")
                    outcome, reason = run_forceguide_episode(
                        executor, follower, keyboard[0], session.capture_frame, session.rig, args.max_episode_steps,
                        session.control_dt, plan_pause_s=plan_pause_s, clock=session.clock, sleep=session.sleep,
                        ctx=ctx, camera_ages=session.camera_ages, clamp_fn=clamp_fn,
                        show_progress=keyboard[0].enabled, label="trial {}{}".format(row["trial_id"], shown),
                        measured_fn=follower.measured)
                except InfrastructureFailure as exc:  # reset failed (ServerChanged included)
                    outcome, reason = "infrastructure", "{}: {}".format(type(exc).__name__, exc)
                finally:
                    if executor is not None:  # the reply in flight is logged with this trial, never applied
                        executor.finish(wait_s=min(1.0, args.infer_timeout), sleep=session.sleep)
            summary = executor.summary() if executor is not None else {"slots": 0, "counts": {}, "bad_frac": 0.0}
            totals["slots"] += summary.get("slots", 0)
            totals["late_any"] += summary.get("counts", {}).get("late_any", 0)
            status = classify(outcome, summary, args.max_bad_refresh_frac)
            flags = protocol_flags(summary, args.max_bad_refresh_frac)
            operator = None
            if status == "labelled" or (status == "infrastructure_failure" and ctx.ticks):
                # Intention to treat (exp 1): the operator enters the outcome of every attempt that ran - protocol-
                # flagged ones and infrastructure failures included - before learning whether it counts.
                if outcome == "infrastructure":  # the reason as the operator may see it (console_reason)
                    print("Trial {} stopped ({}).".format(
                        row["trial_id"], reason if getattr(args, "unblinded", False) and reason else console_reason(reason)))
                else:
                    print("Trial {} ended ({}).".format(row["trial_id"], outcome if reason is None else reason))
                if no_outcome:  # the operator keeps their own record (--no_outcome)
                    operator = {"labelled": False, "success": None, "rerun": False, "stage": None,
                                "failure_type": None, "notes": None}
                else:
                    keyboard[0].close()
                    try:
                        operator = ask_outcome(input_fn)
                    finally:
                        keyboard[0] = keyboard_factory()
            if status == "labelled":
                if not operator["labelled"]:
                    status = "unlabelled"
                elif operator["rerun"]:
                    status = "rerun"
                else:
                    status = "success" if operator["success"] else "failure"
            elif status == "infrastructure_failure":
                # a reason naming refreshes (or a server traceback) would tell the operator the condition's mode;
                # an attempt that ran to its end (reason "key" or None) failed on its refresh slots
                print("Trial {}: infrastructure failure ({}); it will be re-run.".format(
                    row["trial_id"], reason if getattr(args, "unblinded", False) and reason else console_reason(reason, outcome)))
            write_trial(session, row, attempt, status, outcome, reason, operator, summary, ctx, started,
                        time.monotonic() - t0, plan_pause_s, tactile_calibration, preflight, log_dir, episodes_path,
                        codecs=codecs, flags=flags)
            if status == "quit":
                return 0
            if reason is not None and reason.startswith("ServerChanged"):
                print("The server changed during the session (restart?); start a new client session.")
                return 3
            if status in FINAL_STATUSES or status == "unlabelled":
                break
    print("All scheduled trials are done.")
    return 0


def claim_trial_folder(log_dir, trial_id, attempt):
    """(folder, moved_to): trials/<trial>_a<attempt>/ for a new attempt, the path the analysis rebuilds from the
    episodes.jsonl row. A non-empty folder (or a file) of that name is never overwritten: an attempt that crashed
    before its episodes.jsonl row (a resume numbers the next attempt the same) or another schedule / run label forced
    into this log dir. It is moved aside to <name>.prev<k> (first free k >= 1) with a warning; moved_to is None when
    nothing was there."""
    folder = os.path.join(log_dir, "trials", "{}_a{}".format(safe_name(trial_id), attempt))
    moved_to = None
    if os.path.lexists(folder) and (not os.path.isdir(folder) or os.listdir(folder)):
        k = 1
        while os.path.lexists("{}.prev{}".format(folder, k)):
            k += 1
        moved_to = "{}.prev{}".format(folder, k)
        os.rename(folder, moved_to)
        print("WARNING: {} already held data (an attempt that crashed before its episodes.jsonl row, or another run "
              "forced into this log dir); moved it to {} and kept it (trial_folder_moved_to in episodes.jsonl)"
              ".".format(folder, moved_to))
    os.makedirs(folder, exist_ok=True)
    return folder, moved_to


def write_trial(session, row, attempt, status, outcome, reason, operator, summary, ctx, started, duration_s,
                plan_pause_s, tactile_calibration, preflight, log_dir, episodes_path, codecs=None, flags=None):
    """One episodes.jsonl row per attempt. Trial identity: (schedule_sha256, trial_id), attempt, condition, noise_id,
    block; run_label: the label of a run without a schedule file (None for a --schedule campaign). status: success /
    failure (operator) | rerun | unlabelled | infrastructure_failure | quit; operator: the operator's answer (also for
    infrastructure failures that ran); protocol_flags: method / timing outcomes of the refreshes (rejected, late,
    dropped, skipped-in-flight; bad_frac), never a re-run cause; trial_folder_moved_to: where an earlier folder of
    this attempt's name was moved (claim_trial_folder), else None."""
    client, campaign, info = session.client, session.campaign, session.info
    folder, moved_to = claim_trial_folder(log_dir, row["trial_id"], attempt)
    write_jsonl(os.path.join(folder, "ticks.jsonl"), ctx.ticks)
    write_jsonl(os.path.join(folder, "requests.jsonl"), ctx.requests)
    sidecar = os.path.join(folder, "refresh.npz")
    has_sidecar = save_sidecar(sidecar, ctx.sidecar)
    condition = info.condition(row["condition"])
    run = session.run_record or {}
    episode = {"schedule_sha256": run.get("schedule_sha256"), "run_label": run.get("run_label"),
               "trial_id": row["trial_id"], "attempt": attempt,
               "block": row.get("block"), "init_condition_id": row["init_condition_id"], "condition": row["condition"],
               "noise_id": row["noise_id"], "seed": session.args.seed, "session": client.session, "status": status,
               "outcome": outcome, "reason": reason, "operator": operator,
               "protocol_flags": flags if flags is not None else protocol_flags(
                   summary, session.args.max_bad_refresh_frac),
               "started": started, "duration_s": duration_s, "refresh": summary, "plan_pause_s": plan_pause_s,
               "plan_pause_source": run.get("plan_pause_source"), "preflight_check": run.get("preflight_check"),
               "lead": campaign.lead, "refresh_every": campaign.refresh_every, "policy_hz": campaign.policy_hz,
               "refresh_timeout_s": session.args.refresh_timeout, "mode": condition.mode,
               "condition_fingerprint": condition.fingerprint, "server_instance_id": info.server_instance_id,
               "server_git_rev": info.git_rev, "client_git_rev": git_rev(),
               "tactile_payload": {"keys": list(ctx.payload.keys), "proxy": ctx.payload.proxy,
                                   "window": ctx.payload.window, "stride": ctx.payload.stride,
                                   "gap": ctx.payload.gap, "background": ctx.payload.background},
               "tactile_calibration": tactile_calibration, "tactile_codecs": codecs,
               "preflight_required_lead": (preflight or {}).get("required_lead"),
               "logs": {"ticks": os.path.join(folder, "ticks.jsonl"),
                        "requests": os.path.join(folder, "requests.jsonl"),
                        "refresh_npz": sidecar if has_sidecar else None},
               "trial_folder_moved_to": moved_to, "metadata": info.raw}
    episode.update(trial_stats(ctx))
    write_jsonl(episodes_path, [episode], mode="a")
    return episode


if __name__ == "__main__":
    sys.exit(main())
