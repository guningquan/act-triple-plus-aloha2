"""vtla_zmq_client.py against a fake REP server speaking serve_zmq.py's protocol (no GPU, no robot)."""

import threading
import time

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")
msgpack = pytest.importorskip("msgpack")
cv2 = pytest.importorskip("cv2")

import vtla_zmq_client as vzc  # noqa: E402


def decode(buf):
    """serve_zmq.py::_decode_image: imdecode (BGR) -> RGB."""
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class FakeServer:
    def __init__(self, handler, n_requests):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind("tcp://127.0.0.1:*")
        self.port = int(self.sock.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[1])
        self.handler, self.requests = handler, []
        self.thread = threading.Thread(target=self._serve, args=(n_requests,), daemon=True)
        self.thread.start()

    def _serve(self, n):
        for _ in range(n):
            msg = msgpack.unpackb(self.sock.recv(), raw=False)
            self.requests.append(msg)
            self.sock.send(msgpack.packb(self.handler(msg), use_bin_type=True))

    def close(self):
        self.thread.join(5)
        self.sock.close(0)
        self.ctx.term()


def rgb(seed, h=40, w=30):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def handler(msg):
    if msg["cmd"] == "predict":
        return {"status": "ok", "actions": np.arange(50 * 32, dtype=np.float64).reshape(50, 32).tolist(),
                "infer_time_ms": 123.0}
    if msg["cmd"] == "seg_step":
        return {"status": "ok", "k": msg["k"], "p": msg["p"], "replanned": False, "fresh": msg["k"] >= 10}
    if msg["cmd"] == "reset":
        return {"status": "ok"}
    return {"status": "error", "message": "unknown cmd"}


def test_predict_wire_format_and_lossless_png():
    server = FakeServer(handler, 3)
    client = vzc.VtlaZmqClient("127.0.0.1", server.port, timeout_s=5)
    cams = {c: rgb(i) for i, c in enumerate(vzc.CAM_WIRE_KEYS)}
    tacs = {v: rgb(10 + i, 70, 40) for i, v in enumerate(vzc.TAC_VIEWS)}
    hist = {8: {v: rgb(20 + i, 70, 40) for i, v in enumerate(vzc.TAC_VIEWS)}}
    flat = {v: np.full((70, 40, 3), (200, 30, 60), np.uint8) for v in vzc.TAC_VIEWS}   # JPEG-friendly
    try:
        client.reset()
        actions = client.predict(np.arange(14), "plug it in", cams, tacs, hist)
        reply = client.seg_step(10, 10, flat, hist)
    finally:
        client.close()
        server.close()
    assert actions.shape == (50, 14) and actions[1, 0] == 32 and client.last_infer_time_ms == 123.0
    req = server.requests[1]
    assert req["cmd"] == "predict" and req["prompt"] == "plug it in"
    assert len(req["state"]) == 32 and req["state"][:14] == list(range(14)) and not any(req["state"][14:])
    for cam, key in vzc.CAM_WIRE_KEYS.items():
        np.testing.assert_array_equal(decode(req[key]), cams[cam])     # PNG: exact, colour order kept
    for v in vzc.TAC_VIEWS:
        np.testing.assert_array_equal(decode(req["observation/" + v]), tacs[v])
        np.testing.assert_array_equal(decode(req["observation/{}_hist8".format(v)]), hist[8][v])
    probe = server.requests[2]
    assert probe["cmd"] == "seg_step" and probe["k"] == 10 and probe["force"] is False and reply["fresh"] is True
    assert "observation/left_wrist_left_tactile_hist8" in probe
    jpg = decode(probe["observation/left_wrist_left_tactile"]).astype(int)   # JPEG for probes: lossy, RGB kept
    # probes go at the model size (224 x 224, letterboxed like the server's resize): 70 x 40 -> 224 x 128 + 48 px bars
    assert jpg.shape == (224, 224, 3)
    assert np.abs(jpg[20:204, 60:164] - np.array([200, 30, 60])).max() <= 4
    assert jpg[:, :40].max() <= 4 and jpg[:, -40:].max() <= 4


def test_server_error_and_missing_views_raise():
    server = FakeServer(lambda m: {"status": "error", "message": "boom"}, 1)
    client = vzc.VtlaZmqClient("127.0.0.1", server.port, timeout_s=5)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            client.reset()
        with pytest.raises(ValueError, match="tactile views missing"):
            client.seg_step(1, 1, {"left_wrist_left_tactile": rgb(0)})
    finally:
        client.close()
        server.close()


def test_timeout_recreates_the_socket():
    def slow_first(msg, state={"n": 0}):
        state["n"] += 1
        if state["n"] == 1:
            time.sleep(1.0)                 # longer than the client's timeout
        return {"status": "ok"}

    server = FakeServer(slow_first, 2)
    client = vzc.VtlaZmqClient("127.0.0.1", server.port, timeout_s=0.3)
    try:
        with pytest.raises(TimeoutError):
            client.reset()
        time.sleep(1.0)
        client.reset()                      # a fresh REQ socket: the next call works
    finally:
        client.close()
        server.close()
    assert len(server.requests) == 2


# ----------------------------------------------------------------------------------------------- prepare / refine

def refine_handler(msg):
    """serve_zmq.py's prepare / refine protocol (section 6 of the refine spec)."""
    if msg["cmd"] == "prepare":
        return {"status": "ok", "actions": np.arange(50 * 32, dtype=np.float64).reshape(50, 32).tolist(),
                "plan": 7, "infer_time_ms": 456.0}
    if msg["cmd"] == "refine":
        base = {"status": "ok", "plan": msg["plan"], "m": msg["m"], "surprise": [0.1] * 4, "cusum": [0.0] * 4,
                "pad": None, "e": 0.0, "delta": 2.0, "delta_max": 0.01, "saturation": 0.0, "n_reread": 0,
                "ms": 30.0, "infer_time_ms": 31.0}
        if msg["plan"] != 7:
            return dict(base, stale=True, fresh=False)
        if msg["m"] == 13:
            return dict(base, stale=False, fresh=True, pad=2)
        actions = np.full((50, 32), 0.5)
        if msg["m"] == 11:
            actions[20, 3] = np.nan
        return dict(base, stale=False, fresh=False, actions=actions.tolist())
    if msg["cmd"] == "reset":
        return {"status": "ok"}
    return {"status": "error", "message": "unknown cmd"}


def test_prepare_and_refine_wire_format():
    server = FakeServer(refine_handler, 6)
    client = vzc.VtlaZmqClient("127.0.0.1", server.port, timeout_s=5)
    cams = {c: rgb(i) for i, c in enumerate(vzc.CAM_WIRE_KEYS)}
    tacs = {v: rgb(10 + i, 70, 40) for i, v in enumerate(vzc.TAC_VIEWS)}
    hist = {8: {v: rgb(20 + i, 70, 40) for i, v in enumerate(vzc.TAC_VIEWS)}}
    flat = {v: np.full((70, 40, 3), (200, 30, 60), np.uint8) for v in vzc.TAC_VIEWS}
    executed = np.random.default_rng(0).normal(size=(10, 14))
    try:
        actions, plan = client.prepare(np.arange(14), "plug it in", cams, tacs, hist)
        reply = client.refine(plan, 10, executed, flat, hist, state=np.arange(14) * 0.5)
        stale = client.refine(3, 12, np.zeros((12, 14)), flat)
        fresh = client.refine(plan, 13, np.zeros((13, 14)), flat, force=True)
        with pytest.raises(RuntimeError, match="bad actions"):
            client.refine(plan, 11, np.zeros((11, 14)), flat)
        with pytest.raises(ValueError, match="executed must be"):
            client.refine(plan, 4, np.zeros((3, 14)), flat)       # rejected before anything is sent
        client.reset()
    finally:
        client.close()
        server.close()
    assert actions.shape == (50, 14) and plan == 7 and isinstance(plan, int) and client.last_infer_time_ms == 456.0
    req = server.requests[0]
    assert req["cmd"] == "prepare" and req["prompt"] == "plug it in" and len(req["state"]) == 32
    for cam, key in vzc.CAM_WIRE_KEYS.items():
        np.testing.assert_array_equal(decode(req[key]), cams[cam])     # prepare = predict's PNG observation
    np.testing.assert_array_equal(decode(req["observation/left_wrist_left_tactile_hist8"]),
                                  hist[8]["left_wrist_left_tactile"])
    ref = server.requests[1]
    assert ref["cmd"] == "refine" and ref["plan"] == 7 and ref["m"] == 10 and "force" not in ref
    assert np.asarray(ref["executed"]).shape == (10, 14)
    np.testing.assert_allclose(ref["executed"], executed)               # float64 on the wire, exact
    assert len(ref["state"]) == 32 and ref["state"][:14] == pytest.approx(list(np.arange(14) * 0.5))
    assert not any(key in ref for key in vzc.CAM_WIRE_KEYS.values())    # touch only
    for v in vzc.TAC_VIEWS:
        jpg = decode(ref["observation/" + v]).astype(int)               # JPEG at the model size by default
        assert jpg.shape == (224, 224, 3)
        assert np.abs(jpg[20:204, 60:164] - np.array([200, 30, 60])).max() <= 4
        assert decode(ref["observation/{}_hist8".format(v)]).shape == (224, 224, 3)
    assert reply["actions"].shape == (50, 14) and (reply["actions"] == 0.5).all()
    assert reply["stale"] is False and reply["fresh"] is False and reply["surprise"] == [0.1] * 4
    assert stale["stale"] is True and stale["actions"] is None and "state" not in server.requests[2]
    assert fresh["fresh"] is True and fresh["actions"] is None and fresh["pad"] == 2
    assert server.requests[3]["force"] is True
    assert [r["cmd"] for r in server.requests] == ["prepare", "refine", "refine", "refine", "refine", "reset"]


def test_prepare_without_plan_id_raises():
    server = FakeServer(handler, 1)          # the old server answers prepare as an unknown command
    client = vzc.VtlaZmqClient("127.0.0.1", server.port, timeout_s=5)
    cams = {c: rgb(i) for i, c in enumerate(vzc.CAM_WIRE_KEYS)}
    tacs = {v: rgb(10 + i) for i, v in enumerate(vzc.TAC_VIEWS)}
    try:
        with pytest.raises(RuntimeError, match="unknown cmd"):
            client.prepare(np.zeros(14), "", cams, tacs)
    finally:
        client.close()
        server.close()

    server = FakeServer(lambda m: {"status": "ok", "actions": np.zeros((50, 32)).tolist(), "infer_time_ms": 1.0}, 1)
    client = vzc.VtlaZmqClient("127.0.0.1", server.port, timeout_s=5)
    try:
        with pytest.raises(RuntimeError, match="no corrector"):
            client.prepare(np.zeros(14), "", cams, tacs)
    finally:
        client.close()
        server.close()


def test_refine_per_call_timeout_then_default_restored():
    def slow(msg, state={"n": 0}):
        state["n"] += 1
        if state["n"] == 1:
            time.sleep(1.0)                  # the first refine: longer than its 0.2 s timeout
        if state["n"] == 4:
            time.sleep(0.6)                  # a reset slower than the refine timeout, within the default 5 s
        if msg["cmd"] == "refine":
            return {"status": "ok", "plan": msg["plan"], "m": msg["m"], "stale": True, "fresh": False}
        return {"status": "ok"}

    server = FakeServer(slow, 4)
    client = vzc.VtlaZmqClient("127.0.0.1", server.port, timeout_s=5)
    tacs = {v: rgb(i) for i, v in enumerate(vzc.TAC_VIEWS)}
    try:
        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match="0.2 s"):
            client.refine(1, 2, np.zeros((2, 14)), tacs, timeout_s=0.2)
        assert time.monotonic() - t0 < 0.9
        time.sleep(1.0)
        client.reset()                                               # a fresh socket works
        assert client.refine(1, 3, np.zeros((3, 14)), tacs, timeout_s=0.5)["stale"] is True
        client.reset()                                               # 0.6 s: the default timeout is back
    finally:
        client.close()
        server.close()
    assert len(server.requests) == 4


# ------------------------------------------------------------------------------------- model-size probe frames
def _reference_resize():
    """The server's n0vtla_client.image_tools (what ResizeImages applies), loaded from the fuse tree if present
    ($N0VTLA_FUSE_ROOT, else the server's default checkout)."""
    import importlib.util
    import os
    roots = [r for r in (os.environ.get("N0VTLA_FUSE_ROOT"), "/mnt/ssd2/guningquan/Programs_server/N0-VTLA-fuse") if r]
    for root in roots:
        path = root + "/n0vtla_client/image_tools.py"
        try:
            spec = importlib.util.spec_from_file_location("ref_image_tools", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.resize_with_pad
        except (FileNotFoundError, OSError):
            continue
    pytest.skip("no N0-VTLA tree with n0vtla_client/image_tools.py on this machine")


def test_resize_with_pad_matches_the_server_resize_bit_for_bit():
    ref = _reference_resize()
    rs = np.random.RandomState(0)
    for shape in ((700, 400, 3), (400, 700, 3), (480, 640, 3), (70, 40, 3), (225, 223, 3)):
        img = rs.randint(0, 256, shape).astype(np.uint8)
        np.testing.assert_array_equal(vzc.resize_with_pad(img, 224, 224), ref(img, 224, 224))
    model_size = rs.randint(0, 256, (224, 224, 3)).astype(np.uint8)
    assert vzc.resize_with_pad(model_size, 224, 224) is model_size          # already at the size: untouched


def test_probe_and_refine_frames_native_on_request():
    tacs = {v: rgb(10 + i, 70, 40) for i, v in enumerate(vzc.TAC_VIEWS)}
    requests = []
    for handler_fn, call in ((handler, lambda c: c.seg_step(3, 3, tacs, codec="png", tactile_size=None)),
                             (refine_handler, lambda c: c.refine(7, 2, np.zeros((2, 14)), tacs, codec="png",
                                                                 tactile_size=None))):
        server = FakeServer(handler_fn, 1)
        client = vzc.VtlaZmqClient("127.0.0.1", server.port, timeout_s=5)
        try:
            call(client)
        finally:
            client.close()
            server.close()
        requests += server.requests
    assert len(requests) == 2
    for req in requests:
        for v in vzc.TAC_VIEWS:
            np.testing.assert_array_equal(decode(req["observation/" + v]), tacs[v])   # native, lossless


def test_resized_probe_frame_equals_the_server_resize_of_the_native_frame():
    ref = _reference_resize()
    server = FakeServer(refine_handler, 1)
    client = vzc.VtlaZmqClient("127.0.0.1", server.port, timeout_s=5)
    tacs = {v: rgb(30 + i, 700, 400) for i, v in enumerate(vzc.TAC_VIEWS)}
    try:
        client.refine(7, 2, np.zeros((2, 14)), tacs, codec="png")             # default model size, lossless codec
    finally:
        client.close()
        server.close()
    for v in vzc.TAC_VIEWS:
        np.testing.assert_array_equal(decode(server.requests[0]["observation/" + v]), ref(tacs[v], 224, 224))
