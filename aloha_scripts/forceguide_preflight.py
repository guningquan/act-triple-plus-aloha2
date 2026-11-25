#!/usr/bin/env python3
"""Before a ForceGuide campaign: do the live tactile sensors match the training data (which sensor feeds which key,
and in which colour order), and does the VTLA gripper state read ~1.0 at home?

Reference: the npz written by the forceguide repo's scripts/forceguide/export_tactile_reference.py (export it with
--config pi05_board_insertion_tactile4 so that all four sensors are in it: the identity check can only catch swaps
among the keys the npz holds). Layout (what load_reference reads; any other entry is ignored):
    meta_json                   0-d str, JSON: episodes, image_bgr, tactile_bgr, camera_color ("rgb": the camera frames
                                are colour-corrected, i.e. swapped from the dataset order when image_bgr), tactile_color
                                ("rgb"), prepared_spec (else spec): the TactileSpec JSON of tactile_prepared_<i>
    tactile_keys, camera_keys   str [S], str [C]: dataset features observation.images.<name>
    episodes                    int [E]: the dataset episode of each row
    tactile_native_<i>          uint8 [E, H, W, 3] frame 0 of tactile key i, RGB (after the dataset BGR -> RGB swap)
    tactile_prepared_<i>        uint8 [E, 224, 224, 3] tactile_native_<i> prepared per prepared_spec
    camera_rgb_<j>              uint8 [E, H, W, 3] frame 0 of camera j in camera_color order
--reference_episode picks the row by its dataset episode (default: the first row). The older single-frame layout is
still read: '<key>/native' [H, W, 3] (tactile RGB; cameras in the dataset order unless meta camera_bgr_swapped),
'<key>/prepared', 'spec' and 'meta' (0-d JSON str; 'spec' / 'tactile', 'camera_bgr_swapped', 'serials').
Tactile keys are recognised by their last dotted component being a sensor name (right_gripper_inner, ...), cameras
by 'cam_'. The TactileSpec comes from --condition on a running serve_forceguide, else the npz, else the CLI flags.
The reference natives are prepared with that same spec before the identity check (no live codec: the dataset frames
already carry the recorder's codec), so a CPM's resize / rotate (Sparsh's stretch_pil, a rotated sensor) cannot make
a correctly wired rig fail; the npz's own prepared frames only serve a bit-parity note when their spec is the same.

    python3 aloha_scripts/forceguide_preflight.py --reference tactile_reference.npz --out preflight_sensors.json
    python3 aloha_scripts/forceguide_preflight.py --reference ref.npz --host 127.0.0.1 --condition tg_real \
        --reference_episode 25

Checks (robot homed, grippers open, nothing touching the gels):
  colour order   per tactile key: the live frame's per-channel means are closer to the reference frame's than the
                 R/B-swapped live means (the gels' R and B differ by ~80-100 grey levels, so a wrong dataset BGR flag
                 is unmistakable); inconclusive when the two distances differ by less than --colour_margin
  identity       normalized correlation of the live prepared frame (live_codec, rotate, resize per the spec) with every
                 reference key's prepared frame: the intended key must be the best match (by --min_margin)
  cameras        the same colour-order test for the cameras after the image_bgr swap the client applies (advisory
                 unless decisive: a scene's R/B balance is usually much weaker than a gel's)
  gripper        VTLA P0s: FollowerCommander.measured() grippers within --gripper_tolerance of 1.0 at home
  publisher      image_codec must be raw; the sensor serials from /tactile/runtime_config are recorded and, if
                 --expected_serial (or the npz meta 'serials') names them, must match (the training data carries no
                 serials, so the image identity check above is the decisive one)
Writes a JSON report; exit code 0 only if every check passes.
"""

import argparse
import json
import logging
import sys
import time

import numpy as np

from forceguide_client import (KNOWN_SENSORS, ServerInfo, TactileSpec, prepare_live_tactile, prepare_tactile,
                               sensor_for_key, to_jsonable)

CAMERA_PREFIX = "cam_"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="npz from export_tactile_reference.py")
    parser.add_argument("--reference_episode", type=int, default=None,
                        help="Dataset episode of the npz row to compare with (default: its first row)")
    parser.add_argument("--keys", nargs="+", default=None, help="Tactile keys to check (default: all in the npz)")
    parser.add_argument("--out", default=None, help="JSON report path")
    parser.add_argument("--host", default=None, help="serve_forceguide host, to read the condition's spec")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--condition", default=None, help="Condition whose TactileSpec prepares the live frames")
    parser.add_argument("--resize", default=None, choices=("pad_pil", "stretch_pil"))
    parser.add_argument("--rotate", type=int, default=None)
    parser.add_argument("--live_codec", default=None, choices=("none", "yuv420"))
    parser.add_argument("--image_bgr", default="auto", choices=("auto", "true", "false"),
                        help="Camera swap the client applies (auto: server metadata, else the npz's image_bgr, "
                             "else true)")
    parser.add_argument("--state_gripper", default="auto", choices=("auto", "vtla", "realenv"))
    parser.add_argument("--frames", type=int, default=5, help="Live frames averaged per sensor")
    parser.add_argument("--min_margin", type=float, default=0.0, help="Best minus second-best correlation")
    parser.add_argument("--colour_margin", type=float, default=3.0, help="Grey levels for a decisive colour test")
    parser.add_argument("--gripper_tolerance", type=float, default=0.05)
    parser.add_argument("--tactile_sensor", action="append", default=[], metavar="KEY=SENSOR")
    parser.add_argument("--expected_serial", action="append", default=[], metavar="SENSOR=SERIAL",
                        help="Serial the publisher must report for a sensor (/tactile/runtime_config)")
    parser.add_argument("--tactile_wait", type=float, default=10.0)
    parser.add_argument("--calibrate", action="store_true", help="Re-zero the sensors (/tactile/calibrate) first")
    parser.add_argument("--no_robot", action="store_true", help="Tactile checks only (no homing, cameras, gripper)")
    parser.add_argument("--no_home", action="store_true", help="Do not home the followers first")
    parser.add_argument("--no_sleep", action="store_true",
                        help="Leave the followers homed with torque on at the end (default: grippers open, followers "
                             "to the sleep pose, torque off, as run_forceguide_policy.py does)")
    return parser


# --------------------------------------------------------------------------------------------
# Pure checks
# --------------------------------------------------------------------------------------------

def _decode_text(value):
    value = np.asarray(value)
    if value.dtype.kind in ("U", "S", "O") and value.size == 1:
        text = value.reshape(-1)[0]
        return text.decode() if isinstance(text, bytes) else str(text)
    return None


def _json_value(value):
    """A JSON object stored as a dict or as a JSON string (the exporter writes TactileSpec.to_json())."""
    if isinstance(value, str):
        return json.loads(value) if value.strip() else None
    return value


def _load_export_layout(data, path, episode):
    """export_tactile_reference.py's layout: one row of the [E, ...] stacks (see the module docstring)."""
    files = set(data.files)
    meta = json.loads(_decode_text(data["meta_json"]) or "{}") if "meta_json" in files else {}
    if "episodes" in files:
        episodes = [int(e) for e in np.asarray(data["episodes"]).reshape(-1)]
    else:
        episodes = [int(e) for e in meta.get("episodes") or []]
    if episode is None:
        row = 0
    elif int(episode) in episodes:
        row = episodes.index(int(episode))
    else:
        raise ValueError("{} holds episodes {}, not episode {}".format(path, episodes, episode))

    def frame(name):
        array = np.asarray(data[name])
        if array.ndim != 4 or row >= array.shape[0]:
            raise ValueError("{}: {} has shape {}, expected [E, H, W, 3] with row {}".format(path, name, array.shape,
                                                                                            row))
        return array[row]

    def names(entry):
        return [str(name) for name in np.asarray(data[entry]).reshape(-1)] if entry in files else []

    keys = {}
    tactile_swap = str(meta.get("tactile_color") or "rgb").lower() == "bgr"  # the exporter writes RGB
    for i, key in enumerate(names("tactile_keys")):
        entry = {"native": frame("tactile_native_{}".format(i))}
        if "tactile_prepared_{}".format(i) in files:
            entry["prepared"] = frame("tactile_prepared_{}".format(i))
        if tactile_swap:
            entry = {kind: image[..., ::-1] for kind, image in entry.items()}
        keys[key] = entry
    for j, key in enumerate(names("camera_keys")):
        keys[key] = {"native": frame("camera_rgb_{}".format(j))}
    # run_camera_checks compares in the dataset's channel order: the stored camera frames differ from it when they
    # are colour-corrected RGB and the dataset is BGR (image_bgr), or the other way round.
    camera_rgb = str(meta.get("camera_color") or "rgb").lower() == "rgb"  # older exports: camera_rgb_<j> are RGB
    meta["camera_bgr_swapped"] = camera_rgb == bool(meta.get("image_bgr"))
    spec = _json_value(meta.get("prepared_spec")) or _json_value(meta.get("spec")) or _json_value(meta.get("tactile"))
    return {"keys": keys, "spec": spec, "meta": meta, "layout": "export_tactile_reference",
            "episode": episodes[row] if row < len(episodes) else None, "episodes": episodes}


def load_reference(path, episode=None):
    """{"keys": {key: {"native"[, "prepared"]}}, "spec": dict or None, "meta": dict, "layout", "episode"}.

    Reads export_tactile_reference.py's npz (episode: the dataset episode of the row to use, default the first) and
    the older single-frame '<key>/native' layout."""
    keys, meta, spec = {}, {}, None
    with np.load(path, allow_pickle=False) as data:
        if "meta_json" in data.files or "tactile_keys" in data.files:
            return _load_export_layout(data, path, episode)
        if episode is not None:
            raise ValueError("{} holds single frames ('<key>/native'), no episodes to choose from".format(path))
        for name in data.files:
            if name.endswith(("/native", "/prepared")):
                key, kind = name.rsplit("/", 1)
                keys.setdefault(key, {})[kind] = np.asarray(data[name])
            elif name in ("meta", "spec"):
                text = _decode_text(data[name])
                if text:
                    parsed = json.loads(text)
                    if name == "spec":
                        spec = parsed
                    else:
                        meta = parsed
    if spec is None:
        spec = _json_value(meta.get("spec")) or _json_value(meta.get("tactile"))
    return {"keys": keys, "spec": spec, "meta": meta, "layout": "single_frame", "episode": None}


def suffix(key):
    return str(key).rsplit(".", 1)[-1]


def tactile_keys(reference):
    return [key for key in reference["keys"] if suffix(key) in KNOWN_SENSORS]


def camera_keys(reference):
    return [key for key in reference["keys"] if suffix(key).startswith(CAMERA_PREFIX)]


def channel_means(image):
    image = np.asarray(image, dtype=np.float64)
    return image.reshape(-1, image.shape[-1]).mean(axis=0)


def colour_order_check(live, reference, margin=3.0):
    """Are the live frame's channels in the reference's order? Compares per-channel means of the live frame and of
    its R/B swap with the reference's (resolution independent)."""
    live_means = channel_means(live)
    ref_means = channel_means(reference)
    swapped = live_means[::-1]
    distance = float(np.abs(live_means - ref_means).sum())
    swapped_distance = float(np.abs(swapped - ref_means).sum())
    decisive = abs(swapped_distance - distance) >= margin
    verdict = ("ok" if distance < swapped_distance else "swapped") if decisive else "inconclusive"
    return {"live_means": live_means.tolist(), "reference_means": ref_means.tolist(), "distance": distance,
            "swapped_distance": swapped_distance, "verdict": verdict}


def normalized_correlation(a, b, mask=None):
    """Pearson correlation over all pixels and channels (only where mask, an [H, W] bool array, is set)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("cannot correlate shapes {} and {}".format(a.shape, b.shape))
    if mask is not None:
        a, b = a[mask], b[mask]
    a = a.reshape(-1) - a.mean()
    b = b.reshape(-1) - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def content_mask(a, b):
    """Pixels that are not letterbox padding in both images: the zero bars are shared by every sensor and would
    make all correlations look alike."""
    return (np.asarray(a).max(axis=-1) > 0) | (np.asarray(b).max(axis=-1) > 0)


def identity_check(live_prepared, reference_prepared, min_margin=0.0):
    """Per live key: correlation with every reference key; ok when the same key is the best by min_margin."""
    report = {}
    for key, live in live_prepared.items():
        scores = {ref: normalized_correlation(live, image, content_mask(live, image))
                  for ref, image in reference_prepared.items()}
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        best = ranked[0][0] if ranked else None
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float("inf")
        own = scores.get(key)
        report[key] = {"scores": scores, "best": best, "own": own, "margin": margin,
                       "ok": best == key and (len(ranked) < 2 or margin >= min_margin)}
    return report


def serial_check(found, expected):
    """Publisher serials ({sensor: serial} from /tactile/runtime_config) vs the expected ones. ok is None when
    nothing is expected or the publisher reported no serials (unknown, not a failure), else whether all match."""
    expected = dict(expected or {})
    found = dict(found or {})
    per_sensor = {sensor: {"expected": serial, "found": found.get(sensor),
                           "ok": None if not found else found.get(sensor) == serial}
                  for sensor, serial in expected.items()}
    known = [item["ok"] for item in per_sensor.values() if item["ok"] is not None]
    return {"sensors": per_sensor, "ok": (all(known) if known else None)}


def gripper_home_check(measured, arm="both", tolerance=0.05):
    measured = np.asarray(measured, dtype=np.float64)
    indices = {"both": (6, 13), "left": (6,), "right": (13,)}[arm]
    grippers = [float(measured[i]) for i in indices]
    return {"grippers": grippers, "tolerance": tolerance, "ok": all(abs(g - 1.0) <= tolerance for g in grippers)}


def reference_prepared_frames(reference, spec):
    """Every reference tactile key prepared with the resolved spec (rotate, resize; no live codec: the dataset
    frames already carry the recorder's codec), so live and reference frames have the same geometry whatever spec
    the npz's own prepared frames were made with (parity finding 2). Keys without a native frame keep the npz's."""
    out = {}
    for key, entry in reference["keys"].items():
        if suffix(key) not in KNOWN_SENSORS:
            continue
        if "native" in entry:
            out[key] = prepare_tactile(entry["native"], spec)
        elif "prepared" in entry:
            out[key] = entry["prepared"]
    return out


def prepared_parity(reference, spec):
    """Bit parity of this client's prepare_tactile with the npz's prepared frames, where the npz spec has the same
    geometry (resize, rotate) as the resolved one: max |difference| per key (0 = bit-identical). Informational."""
    npz_spec = reference.get("spec") or {}
    same = (npz_spec.get("resize") == spec.resize and int(npz_spec.get("rotate") or 0) % 4 == spec.rotate)
    out = {"same_geometry": bool(same), "npz": {"resize": npz_spec.get("resize"), "rotate": npz_spec.get("rotate")},
           "resolved": {"resize": spec.resize, "rotate": spec.rotate}, "max_abs_diff": {}}
    if same:
        for key, entry in reference["keys"].items():
            if suffix(key) in KNOWN_SENSORS and "native" in entry and "prepared" in entry:
                mine = prepare_tactile(entry["native"], spec)
                if mine.shape == np.shape(entry["prepared"]):
                    out["max_abs_diff"][key] = int(np.abs(mine.astype(np.int32) - entry["prepared"]).max())
    return out


def run_tactile_checks(live_native, reference, spec, keys=None, min_margin=0.0, colour_margin=3.0):
    """live_native: {key: native RGB frame}. Colour order per key, then identity over every reference key (both
    sides prepared with `spec`)."""
    keys = list(keys or live_native)
    colour = {key: colour_order_check(live_native[key], reference["keys"][key]["native"], colour_margin)
              for key in keys if "native" in reference["keys"].get(key, {})}
    live_prepared = {key: prepare_live_tactile(live_native[key], spec) for key in keys}
    identity = identity_check(live_prepared, reference_prepared_frames(reference, spec), min_margin)
    ok = (all(item["verdict"] == "ok" for item in colour.values()) and all(item["ok"] for item in identity.values())
          and len(colour) == len(keys))
    return {"colour": colour, "identity": identity, "keys": keys, "ok": ok,
            "prepared_parity": prepared_parity(reference, spec)}


def run_camera_checks(live_images, reference, image_bgr, margin=3.0):
    """Model-input order of the live cameras (after the client's image_bgr swap) vs the training frames in the
    dataset's order. The npz frames are in the dataset order unless its meta says camera_bgr_swapped (load_reference
    sets it for the exporter's colour-corrected camera_rgb_<j> of a BGR dataset): a live frame equal to the
    reference frame, with the P0's image_bgr, reads "ok"."""
    swapped_reference = bool(reference.get("meta", {}).get("camera_bgr_swapped"))
    report = {}
    for key in camera_keys(reference):
        camera = suffix(key)
        if camera not in live_images or "native" not in reference["keys"][key]:
            continue
        live = np.asarray(live_images[camera])
        model_input = live[..., ::-1] if image_bgr else live
        target = reference["keys"][key]["native"]
        if swapped_reference:
            target = target[..., ::-1]
        report[camera] = colour_order_check(model_input, target, margin)
    ok = all(item["verdict"] != "swapped" for item in report.values())
    return {"cameras": report, "image_bgr": bool(image_bgr), "ok": ok}


def resolve_spec(args, reference, info=None):
    """TactileSpec for the live preparation: --condition (server) > npz spec > CLI flags (default pad_pil)."""
    data = None
    if info is not None and args.condition:
        spec = info.condition(args.condition).tactile
        if spec is None:
            raise SystemExit("condition {!r} has no tactile spec".format(args.condition))
        data = spec.to_dict()
    elif reference.get("spec"):
        data = dict(reference["spec"])
    keys = args.keys or (data or {}).get("keys") or tactile_keys(reference)  # default: the CPM's sensors
    data = dict(data or {}, keys=keys)
    for name in ("resize", "rotate", "live_codec"):
        value = getattr(args, name)
        if value is not None:
            data[name] = value
    data.setdefault("resize", "pad_pil")
    if data["resize"] == "pad":
        raise SystemExit("the reference was prepared with the legacy 'pad' resize; export it with pad_pil/stretch_pil")
    return TactileSpec.from_dict(data)


# --------------------------------------------------------------------------------------------
# Live capture
# --------------------------------------------------------------------------------------------

def capture_live_tactile(sensors, frames=5, timeout_s=10.0, acquire=None, sleep=time.sleep):
    """Average of `frames` distinct live rectify frames per sensor (true RGB), plus their codec and seqs."""
    if acquire is None:
        from tactile_stream import TactileStream

        acquire = TactileStream.acquire
    streams = {sensor: acquire(sensor, prefetch=(("rectify", "rgb"),)) for sensor in sensors}
    out, codecs, seqs = {}, {}, {}
    try:
        deadline = time.monotonic() + timeout_s
        for sensor, stream in streams.items():
            images, seen = [], []
            while len(images) < frames:
                frame = stream.latest()
                if frame is not None and frame.seq not in seen:
                    seen.append(frame.seq)
                    images.append(np.asarray(frame.field("rectify", "rgb"), dtype=np.float32))
                    codecs[sensor] = (frame.meta or {}).get("image_codec")
                elif time.monotonic() > deadline:
                    raise RuntimeError("only {} frames from {} within {:.0f} s".format(len(images), sensor, timeout_s))
                else:
                    sleep(0.01)
            out[sensor] = np.clip(np.rint(np.mean(images, axis=0)), 0, 255).astype(np.uint8)
            seqs[sensor] = seen
    finally:
        for stream in streams.values():
            release = getattr(stream, "release", None)
            if release is not None:
                release()
    return out, codecs, seqs


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    reference = load_reference(args.reference, args.reference_episode)
    overrides = dict(value.split("=", 1) for value in args.tactile_sensor)
    info = None
    if args.host:
        from forceguide_client import ForceGuideConnection

        connection = ForceGuideConnection(args.host, args.port, connect_timeout_s=10.0, name="preflight")
        info = ServerInfo.from_metadata(connection.metadata)
        connection.close()
    spec = resolve_spec(args, reference, info)
    keys = list(spec.keys)
    missing = [key for key in keys if key not in reference["keys"]]
    if missing:
        raise SystemExit("keys {} are not in the reference npz (it has {})".format(missing, sorted(reference["keys"])))
    reference_image_bgr = (reference.get("meta") or {}).get("image_bgr")
    if args.image_bgr != "auto":
        image_bgr = args.image_bgr == "true"
    elif info is not None:
        image_bgr = info.image_bgr
    else:
        image_bgr = bool(reference_image_bgr) if reference_image_bgr is not None else True
    state_gripper = args.state_gripper if args.state_gripper != "auto" else (info.state_gripper if info else "vtla")
    arm = info.arm if info else "both"

    report = {"reference": args.reference, "reference_layout": reference.get("layout"),
              "reference_episode": reference.get("episode"), "reference_spec": reference.get("spec"),
              "reference_image_bgr": reference_image_bgr, "spec": spec.to_dict(), "keys": keys,
              "image_bgr": image_bgr, "state_gripper": state_gripper}
    if reference_image_bgr is not None and bool(reference_image_bgr) != image_bgr:
        logging.warning("the client's image_bgr %s differs from the reference dataset's %s: the camera check judges "
                        "the model input against the reference dataset's channel order", image_bgr,
                        reference_image_bgr)
    import tactile_stream

    env = None
    if args.no_robot:
        import rospy

        rospy.init_node("forceguide_preflight", anonymous=True)
    else:
        from constants import DT, PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, START_ARM_POSE
        from real_env import make_real_env
        from robot_utils import move_arms, move_grippers
        from run_pi_policy import home_followers

        env = make_real_env(init_node=True, setup_robots=True)
        if not args.no_home:
            reset_pose = list(((info.raw if info else {}) or {}).get("reset_pose") or START_ARM_POSE)[:6]
            home_followers((env.puppet_bot_left, env.puppet_bot_right), reset_pose, move_arms, move_grippers,
                           (PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN), DT)
    final_lines = []  # printed after the followers are asleep, so the robot's shutdown output cannot bury them
    try:
        sensors = {key: sensor_for_key(key, overrides) for key in keys}
        if args.calibrate:
            tactile_stream.calibrate_sensors(sorted(set(sensors.values())))
        config = tactile_stream.RuntimeConfig.get().wait(3.0)
        report["runtime_config"] = None if not config else {key: config.get(key) for key in (
            "serials", "calibrated", "calibrated_at", "sdk_version", "settings")}
        expected_serials = dict((reference.get("meta") or {}).get("serials") or {})
        expected_serials.update(dict(value.split("=", 1) for value in args.expected_serial))
        report["serials"] = serial_check((config or {}).get("serials"), expected_serials)
        live, codecs, seqs = capture_live_tactile(sorted(set(sensors.values())), args.frames, args.tactile_wait)
        report["codecs"], report["seqs"] = codecs, seqs
        report["publisher_ok"] = all(codec in (None, "raw") for codec in codecs.values())
        live_native = {key: live[sensor] for key, sensor in sensors.items()}
        report["tactile"] = run_tactile_checks(live_native, reference, spec, keys, args.min_margin, args.colour_margin)
        checks = [report["tactile"]["ok"], report["publisher_ok"], report["serials"]["ok"] is not False]
        if env is not None:
            images = env.get_images()
            report["cameras"] = run_camera_checks(images, reference, image_bgr, args.colour_margin)
            checks.append(report["cameras"]["ok"])
            from hil_devices import FollowerCommander

            measured = FollowerCommander((env.puppet_bot_left, env.puppet_bot_right), max_joint_step=None).measured()
            report["gripper"] = gripper_home_check(measured, arm, args.gripper_tolerance)
            report["gripper"]["applies"] = state_gripper == "vtla"
            if state_gripper == "vtla":
                checks.append(report["gripper"]["ok"])
        report["ok"] = all(checks)
        text = json.dumps(to_jsonable(report), indent=2)
        if args.out:
            with open(args.out, "w") as handle:
                handle.write(text)
        if not args.out:
            final_lines.append(text)
        for key, item in report["tactile"]["identity"].items():
            colour = report["tactile"]["colour"].get(key, {}).get("verdict")
            final_lines.append("{:45s} best match {:45s} corr {:.3f} margin {:.3f} colour {}".format(
                key, str(item["best"]), item["scores"].get(item["best"], float("nan")), item["margin"], colour))
        parity = report["tactile"].get("prepared_parity") or {}
        if parity.get("max_abs_diff"):
            final_lines.append("client preparation vs the npz's prepared frames (same geometry): max |diff| {}".format(
                parity["max_abs_diff"]))
        if "gripper" in report:
            final_lines.append("gripper state at home: {}".format(report["gripper"]["grippers"]))
        final_lines.append("{} (report: {})".format("PASS" if report["ok"] else "FAIL", args.out or "stdout"))
        return 0 if report["ok"] else 1
    finally:
        if env is not None and not args.no_sleep:  # never leave the followers up with torque on
            from sleep_plus import shut_down_puppet_robots

            print("Opening the grippers and moving the followers to the sleep pose...")
            shut_down_puppet_robots()
            print("Followers are at the sleep pose with torque off.")
        for line in final_lines:
            print(line)


if __name__ == "__main__":
    sys.exit(main())
