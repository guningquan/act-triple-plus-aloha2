"""Shared, decode-once access to xense_tactile_publisher's tactile topics.

4arms_tactile_teleop.launch publishes, per sensor, /tactile/<sensor>/raw: a pickled
{"frames": {...}, "meta": {...}} bundle, sent the moment the XENSE SDK delivers each new frame.
Several parts of one process want the same topic (the HDF5 / VTLA recorders, the leader-gripper
force feedback, the policy runners), so instead of each opening its own subscriber and
unpickling every ~1 MB message again, they all share one TactileStream per topic per process:

    stream = TactileStream.acquire("left_gripper_outer")      # or a full topic name
    frame = stream.latest()                                   # TactileFrame or None
    frame.age()                                               # s since the SDK delivered it
    frame.field("rectify", order="bgr")                       # decoded once, cached
    stream.release()

The callback only unpickles; anything heavier (colour conversion, JPEG decoding) happens on
first use and is cached on the frame, or eagerly in the callback thread for the (field, order)
pairs a consumer passes as `prefetch`, so a control loop never pays for it.

Frame age is the data's own age: now - meta["local_read_end"], the host time.time() at which
the publisher's SDK call returned (same clock as this process). Receipt time would hide a
publisher that keeps re-sending an old frame; the data's age cannot.

The publisher's ~image_codec decides how rectify/difference travel: "raw" (uint8 RGB arrays,
the default) or "jpeg" (true-colour JPEG bytes; depth then zlib-compressed float32).
meta["encoding"] says which keys are encoded; decode_field()/TactileFrame.field() hide the
difference, and jpeg_bytes() hands out the JPEG unchanged for recorders that store it as is.

Also here: the latched /tactile/runtime_config topic (RuntimeConfig) and calibrate_sensors(),
which asks the publisher to re-zero every sensor (/tactile/calibrate) and waits for fresh frames.
"""

import pickle
import sys
import threading
import time
import zlib

import cv2
import numpy as np

# One module - hence one stream registry - per process, whichever way it is imported: scripts in
# aloha_scripts/ import it flat ("tactile_stream"), while one_side_teleop.py first tries
# "aloha_scripts.tactile_stream" (when the repo root is importable). Two copies would mean two
# subscriptions per topic, so register this one under both names.
for _alias in ("tactile_stream", "aloha_scripts.tactile_stream"):
    sys.modules.setdefault(_alias, sys.modules[__name__])

RAW_TOPIC = "/tactile/{}/raw"
RUNTIME_CONFIG_TOPIC = "/tactile/runtime_config"
CALIBRATE_SERVICE = "/tactile/calibrate"
COLOUR_FIELDS = ("rectify", "difference")
# Refuse tactile data older than this (by the data's own age). Measured 2026-11-08: the newest
# frame is at most ~35-55 ms old in normal operation, so 0.2 s only trips on a real stall.
DEFAULT_MAX_TACTILE_AGE_S = 0.2


class TactileStale(RuntimeError):
    """A tactile sensor has no frame yet, or its newest frame is older than allowed."""


def topic_for(sensor_or_topic):
    """/tactile/<sensor>/raw for a sensor name; a name starting with "/" is taken as the topic."""
    return sensor_or_topic if sensor_or_topic.startswith("/") else RAW_TOPIC.format(sensor_or_topic)


def decode_field(frames, meta, key, order="rgb"):
    """One field of a /raw payload as a numpy array, whatever image codec the publisher used.

    rectify/difference: uint8 (H, W, 3) in the requested channel order - "rgb" is the raw
    publisher's own order, "bgr" is OpenCV's. A raw RGB array comes back as is for "rgb" (treat it
    as read-only) and as a new array for "bgr". depth: float32 mm. Other fields: as published.
    Returns None when the field is not in the payload.
    """
    value = frames.get(key)
    if value is None:
        return None
    encoding = ((meta or {}).get("encoding") or {}).get(key)
    if encoding is not None:
        codec = encoding.get("codec")
        if codec == "jpeg":
            bgr = cv2.imdecode(np.asarray(value, dtype=np.uint8).reshape(-1), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cannot decode the JPEG in tactile field {}".format(key))
            return bgr if order == "bgr" else np.ascontiguousarray(bgr[..., ::-1])
        if codec == "zlib":
            raw = zlib.decompress(np.asarray(value, dtype=np.uint8).tobytes())
            return np.frombuffer(raw, dtype=np.dtype(encoding.get("dtype", "float32"))).reshape(encoding["shape"])
        raise ValueError("unknown tactile encoding {!r} for field {}".format(codec, key))
    if key in COLOUR_FIELDS and order == "bgr":
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[2] == 3:
            return np.ascontiguousarray(array[..., ::-1])
    return value


class TactileFrame:
    """One /raw message as received. Decoded fields are cached; treat everything as read-only."""

    def __init__(self, frames, meta, recv_time, recv_mono):
        self.frames = frames
        self.meta = meta or {}
        self.recv_time = recv_time   # host time.time() at receipt
        self.recv_mono = recv_mono   # host time.monotonic() at receipt
        self.count = 0               # per-stream message counter, set by the stream
        self._cache = {}
        self._lock = threading.Lock()

    @property
    def read_end(self):
        """Host time.time() at which the publisher's SDK call returned this frame."""
        value = self.meta.get("local_read_end")
        return float(value) if value is not None else self.recv_time

    @property
    def read_start(self):
        value = self.meta.get("local_read_start")
        return float(value) if value is not None else float("nan")

    @property
    def timestamp(self):
        """The SDK's own TimeStamp for this frame (falls back to read_end)."""
        value = self.meta.get("timestamp")
        return float(value) if value is not None else self.read_end

    @property
    def seq(self):
        """The publisher's per-sensor frame counter (falls back to this stream's message count)."""
        value = self.meta.get("seq")
        return int(value) if value is not None else self.count

    def age(self, now=None):
        return (time.time() if now is None else now) - self.read_end

    def keys(self):
        return list(self.frames.keys())

    def encoding(self, key):
        return (self.meta.get("encoding") or {}).get(key)

    def jpeg_bytes(self, key):
        """The field's JPEG bytes (1-D uint8) if it travelled as JPEG, else None."""
        encoding = self.encoding(key)
        if encoding is not None and encoding.get("codec") == "jpeg":
            return np.asarray(self.frames[key], dtype=np.uint8).reshape(-1)
        return None

    def field(self, key, order="rgb"):
        """decode_field() for this frame, computed at most once per (key, order)."""
        cache_key = (key, order if key in COLOUR_FIELDS else None)
        with self._lock:
            if cache_key not in self._cache:
                self._cache[cache_key] = decode_field(self.frames, self.meta, key, order)
            return self._cache[cache_key]


class TactileStream:
    """The newest frame of one /tactile/<sensor>/raw topic, shared by everything in this process."""

    _registry_lock = threading.Lock()
    _streams = {}

    @classmethod
    def acquire(cls, sensor_or_topic, prefetch=()):
        """The process-wide stream for this sensor/topic (created on first use), refcounted.

        prefetch: (field, order) pairs to decode in the callback thread as each frame arrives,
        e.g. (("rectify", "bgr"),) for a video recorder, so its control loop gets them ready-made.
        """
        topic = topic_for(sensor_or_topic)
        with cls._registry_lock:
            stream = cls._streams.get(topic)
            if stream is None:
                stream = cls(topic)
                cls._streams[topic] = stream
            stream._refs += 1
            stream._prefetch.update(tuple(p) for p in prefetch)
        return stream

    @classmethod
    def register(cls, stream):
        """Make an already-built stream (e.g. subscribe=False, for tests/tools) the shared one."""
        with cls._registry_lock:
            cls._streams[stream.topic] = stream

    def __init__(self, topic, subscribe=True):
        self.topic = topic
        self._cond = threading.Condition()
        self._latest = None
        self._count = 0
        self._errors = 0
        self._refs = 0
        self._prefetch = set()
        self._sub = None
        if subscribe:
            import rospy
            from std_msgs.msg import UInt8MultiArray

            self._rospy = rospy
            # A small buff_size queues ~1 MB messages and the consumer falls seconds behind.
            self._sub = rospy.Subscriber(topic, UInt8MultiArray, self._callback, queue_size=1,
                                         buff_size=2 ** 27, tcp_nodelay=True)

    def release(self):
        """Drop one reference; the last one unsubscribes."""
        with TactileStream._registry_lock:
            self._refs -= 1
            if self._refs > 0:
                return
            if TactileStream._streams.get(self.topic) is self:
                del TactileStream._streams[self.topic]
        if self._sub is not None:
            self._sub.unregister()
            self._sub = None

    def _callback(self, msg):
        recv_time, recv_mono = time.time(), time.monotonic()
        try:
            payload = pickle.loads(bytes(msg.data))
        except Exception as exc:  # noqa: BLE001 - drop a malformed message, keep the last good one
            self._errors += 1
            self._rospy.logwarn_throttle(5.0, "tactile_stream: cannot unpickle {}: {}".format(self.topic, exc))
            return
        self.feed(payload, recv_time, recv_mono)

    def feed(self, payload, recv_time=None, recv_mono=None):
        """Take one unpickled payload as the newest frame (the callback's path; tests use it directly)."""
        frame = TactileFrame(payload.get("frames", {}), payload.get("meta", {}),
                             time.time() if recv_time is None else recv_time,
                             time.monotonic() if recv_mono is None else recv_mono)
        for key, order in list(self._prefetch):
            try:
                frame.field(key, order)
            except Exception:  # noqa: BLE001 - the consumer will see the error on its own read
                pass
        with self._cond:
            self._count += 1
            frame.count = self._count
            self._latest = frame
            self._cond.notify_all()
        return frame

    def latest(self):
        with self._cond:
            return self._latest

    @property
    def count(self):
        """Messages received so far."""
        with self._cond:
            return self._count

    def wait_for_frame(self, timeout_s):
        """The newest frame, waiting up to timeout_s for the first one; None if none arrived."""
        return self.wait_newer_than(float("-inf"), timeout_s)

    def wait_newer_than(self, t, timeout_s):
        """Wait for a frame the SDK delivered after host time t; return it, or None on timeout."""
        deadline = time.time() + timeout_s
        with self._cond:
            while self._latest is None or self._latest.read_end <= t:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cond.wait(min(remaining, 0.05))
            return self._latest


class RuntimeConfig:
    """The latched /tactile/runtime_config blob: every sensor's SDK runtime config, exported by the
    publisher right after its last zeroing, plus the SDK version and the publisher's settings."""

    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self, subscribe=True):
        self._cond = threading.Condition()
        self._blob = None
        if subscribe:
            import rospy
            from std_msgs.msg import UInt8MultiArray

            self._sub = rospy.Subscriber(RUNTIME_CONFIG_TOPIC, UInt8MultiArray, self._callback, queue_size=1)

    def _callback(self, msg):
        try:
            blob = pickle.loads(bytes(msg.data))
        except Exception:  # noqa: BLE001
            return
        if isinstance(blob, dict):
            self.feed(blob)

    def feed(self, blob):
        with self._cond:
            self._blob = blob
            self._cond.notify_all()

    def latest(self):
        with self._cond:
            return self._blob

    def wait(self, timeout_s, newer_than=None):
        """The blob, waiting up to timeout_s for one (exported after host time newer_than if given)."""
        deadline = time.time() + timeout_s
        with self._cond:
            while self._blob is None or (newer_than is not None
                                         and float(self._blob.get("exported_at") or 0.0) <= newer_than):
                remaining = deadline - time.time()
                if remaining <= 0:
                    return self._blob if newer_than is None else None
                self._cond.wait(min(remaining, 0.05))
            return self._blob


def calibrate_sensors(sensors, service_timeout_s=5.0, fresh_timeout_s=3.0):
    """Re-zero every tactile sensor through the publisher, then wait for fresh frames.

    Nothing may touch the gels while this runs. `sensors` are the sensor names (or topics) whose
    fresh frames to wait for. Returns the runtime-config blob that zeroing exported (or None if it
    did not arrive in time). Raises RuntimeError if the service is missing or reports failure.
    """
    import rospy
    from std_srvs.srv import Trigger

    config = RuntimeConfig.get()
    streams = [TactileStream.acquire(sensor) for sensor in sensors]
    try:
        try:
            rospy.wait_for_service(CALIBRATE_SERVICE, timeout=service_timeout_s)
        except Exception as exc:  # noqa: BLE001 - rospy raises ROSException on timeout
            raise RuntimeError("{} is not available - is the tactile publisher (4arms_tactile_teleop.launch) "
                               "running? ({})".format(CALIBRATE_SERVICE, exc))
        t_call = time.time()
        response = rospy.ServiceProxy(CALIBRATE_SERVICE, Trigger)()
        if not response.success:
            raise RuntimeError("tactile zeroing failed: {}".format(response.message))
        t_done = time.time()
        for stream in streams:
            if stream.wait_newer_than(t_done, fresh_timeout_s) is None:
                raise RuntimeError("no fresh frame on {} within {:.1f} s after zeroing".format(
                    stream.topic, fresh_timeout_s))
        blob = config.wait(fresh_timeout_s, newer_than=t_call)
        if blob is None:
            print("Warning: the sensors were zeroed, but no new {} arrived within {:.0f} s; this episode is "
                  "saved without the tactile runtime config.".format(RUNTIME_CONFIG_TOPIC, fresh_timeout_s))
        return blob
    finally:
        for stream in streams:
            stream.release()


def current_runtime_config(timeout_s=3.0):
    """The latched runtime-config blob from the publisher's last zeroing, for an episode recorded
    without re-zeroing (--no_tactile_calibration); None, with a warning, if none arrives in time."""
    blob = RuntimeConfig.get().wait(timeout_s)
    if blob is None:
        print("Warning: no {} within {:.0f} s (is the tactile publisher up to date?); this episode is "
              "saved without the tactile runtime config.".format(RUNTIME_CONFIG_TOPIC, timeout_s))
    return blob
