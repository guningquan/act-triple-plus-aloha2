"""Hardware-free test setup: aloha_scripts on sys.path, ROS / Interbotix replaced by stubs.

Run with the robot's Python 3.8 (needs pytest + msgpack + websockets):
    cd act-triple-plus-aloha2 && python3 -m pytest aloha_scripts/tests -q
"""

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


class JointSingleCommand:
    def __init__(self, name="", cmd=0.0):
        self.name = name
        self.cmd = cmd


class _NoHardware:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("tests must not construct real Interbotix robots")


_stub("rospy", get_time=lambda: 0.0, logwarn_throttle=lambda *a, **k: None, Subscriber=_NoHardware,
      is_shutdown=lambda: False, init_node=lambda *a, **k: None, sleep=lambda *a: None)
_stub("interbotix_xs_msgs")
_stub("interbotix_xs_msgs.msg", JointSingleCommand=JointSingleCommand)
_stub("interbotix_xs_modules")
_stub("interbotix_xs_modules.arm", InterbotixManipulatorXS=_NoHardware)
_stub("cv_bridge", CvBridge=_NoHardware)
_stub("sensor_msgs")
_stub("sensor_msgs.msg", Image=object, JointState=object)
_stub("std_msgs")
_stub("std_msgs.msg", UInt8MultiArray=object)
