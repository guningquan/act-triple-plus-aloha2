import numpy as np
import time
try:
    from aloha_scripts.constants import DT
except ImportError:
    from constants import DT
from interbotix_xs_msgs.msg import JointSingleCommand

import IPython
e = IPython.embed

class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.is_debug = is_debug
        self.bridge = CvBridge()

        # Add "gel" to our list of cameras
        self.camera_names = [
            'cam_high',
            'cam_low',
            'cam_left_wrist',
            'cam_right_wrist',
            # 'gel',   # @gnq important !
        ]

        if init_node:
            rospy.init_node('image_recorder', anonymous=True)

        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)

            # Select callback based on camera name
            if cam_name == 'cam_high':
                callback_func = self.image_cb_cam_high
                topic_name = "/usb_cam_high/image_raw"

            elif cam_name == 'cam_low':
                callback_func = self.image_cb_cam_low
                topic_name = "/usb_cam_low/image_raw"

            elif cam_name == 'cam_left_wrist':
                callback_func = self.image_cb_cam_left_wrist
                topic_name = "/usb_cam_left_wrist/image_raw"

            elif cam_name == 'cam_right_wrist':
                callback_func = self.image_cb_cam_right_wrist
                topic_name = "/usb_cam_right_wrist/image_raw"

            elif cam_name == 'gel':
                callback_func = self.image_cb_gel
                # Make sure this matches the topic used in digit.py
                topic_name = "/gel/camera/image_color"

            else:
                # If you add more names later, handle them here
                raise NotImplementedError

            # Subscribe to the correct topic with the selected callback
            rospy.Subscriber(topic_name, Image, callback_func)

            # If debug, store the last 50 timestamps for frequency diagnostics
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))

        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        setattr(self, f'{cam_name}_image', cv_image)
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)

        if self.is_debug:
            ts_list = getattr(self, f'{cam_name}_timestamps')
            ts_list.append(data.header.stamp.secs + data.header.stamp.nsecs * 1e-9)

    def image_cb_cam_high(self, data):
        return self.image_cb('cam_high', data)

    def image_cb_cam_low(self, data):
        return self.image_cb('cam_low', data)

    def image_cb_cam_left_wrist(self, data):
        return self.image_cb('cam_left_wrist', data)

    def image_cb_cam_right_wrist(self, data):
        return self.image_cb('cam_right_wrist', data)

    def image_cb_gel(self, data):
        return self.image_cb('gel', data)

    def get_images(self):
        image_dict = {}
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

    def print_diagnostics(self):
        def dt_helper(list_of_timestamps):
            arr = np.array(list_of_timestamps)
            diff = arr[1:] - arr[:-1]
            return np.mean(diff)

        for cam_name in self.camera_names:
            ts_list = getattr(self, f'{cam_name}_timestamps', [])
            if len(ts_list) > 1:
                image_freq = 1 / dt_helper(ts_list)
                print(f'{cam_name} frequency: {image_freq:.2f} Hz')
            else:
                print(f'{cam_name} no timestamps recorded yet.')
        print()

class Recorder:
    def __init__(self, side, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from sensor_msgs.msg import JointState
        from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand

        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.is_debug = is_debug

        if init_node:
            rospy.init_node('recorder', anonymous=True)
        rospy.Subscriber(f"/puppet_{side}/joint_states", JointState, self.puppet_state_cb)
        rospy.Subscriber(f"/puppet_{side}/commands/joint_group", JointGroupCommand, self.puppet_arm_commands_cb)
        rospy.Subscriber(f"/puppet_{side}/commands/joint_single", JointSingleCommand, self.puppet_gripper_commands_cb)
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def puppet_state_cb(self, data):
        self.qpos = data.position
        self.qvel = data.velocity
        self.effort = data.effort
        self.data = data
        if self.is_debug:
            self.joint_timestamps.append(time.time())

    def puppet_arm_commands_cb(self, data):
        self.arm_command = data.cmd
        if self.is_debug:
            self.arm_command_timestamps.append(time.time())

    def puppet_gripper_commands_cb(self, data):
        self.gripper_command = data.cmd
        if self.is_debug:
            self.gripper_command_timestamps.append(time.time())

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

        print(f'{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n')

def get_arm_joint_positions(bot):
    return bot.arm.core.joint_states.position[:6]

def get_arm_gripper_positions(bot):
    joint_position = bot.gripper.core.joint_states.position[6]
    return joint_position

def move_arms(bot_list, target_pose_list, move_time=1):
    """Move every bot in bot_list to its target in target_pose_list, in lockstep, over move_time.

    One trajectory command per bot, with moving_time=move_time - the servo's own time-based
    Profile_Velocity/Profile_Acceleration already interpolates smoothly from wherever the arm
    currently is, so there is no need to also step through a python-side linspace.

    The previous implementation did both at once and they fought each other: it published a new
    micro-step target every DT (0.02s) but never told the servo to expect anything that fast, so
    each arm's trajectory-time registers stayed at whatever InterbotixArmXSInterface.__init__
    last set them to - unconditionally 2.0s, since every InterbotixManipulatorXS(...) call
    re-applies its own moving_time default at construction, regardless of what any earlier
    process left there. Every 20ms the servo would restart a fresh "reach this in 2 seconds"
    plan toward a target only 1/num_steps of the way along, so it barely built up speed and the
    arm was still well short of the destination when move_time elapsed. A caller that then waits
    only ~move_time (sleep_plus.py's shutdown path did) before cutting torque could catch the arm
    mid-trajectory and let it drop under its own weight - that is what this rewrite fixes: by
    setting moving_time explicitly, the register always matches what every caller here already
    assumes, so sleeping move_time afterwards is actually accurate.

    One more gotcha this glosses over on your behalf: set_joint_positions() runs
    check_joint_limits() BEFORE publish_positions() (which is the call that actually applies a
    new moving_time via set_trajectory_time()) - so its velocity sanity check
    (|target - last_commanded| / bot.arm.moving_time) reads whatever moving_time was left over
    from the arm's PREVIOUS use, not the one being requested now. A puppet arm coming straight
    out of teleop has moving_time=DT=0.02s (set once in prep_robots() for responsive tracking,
    then never touched again by the per-tick publish_positions() calls, which pass no
    moving_time of their own) - so a completely ordinary move divided by that stale 0.02s looks
    like tens of rad/s, blows past every joint's real velocity limit (~pi rad/s here), and
    set_joint_positions() silently returns False. The leader/master arm never has this problem
    (nothing publish_positions()es it during teleop - the human moves it directly), which is why
    a shutdown can visibly move the leader while the follower just sits there with only a
    one-line, easy-to-miss warning explaining why. Calling set_trajectory_time() by itself first
    updates moving_time immediately, before check_joint_limits() ever runs, so its velocity
    check sees the real, intended duration.
    """
    accel_time = max(move_time / 4.0, 0.05)
    for bot, target in zip(bot_list, target_pose_list):
        bot.arm.set_trajectory_time(moving_time=move_time, accel_time=accel_time)
        if not bot.arm.set_joint_positions(list(target), moving_time=move_time, accel_time=accel_time, blocking=False):
            print("Warning: {} arm target rejected (would exceed joint limits); holding position."
                  .format(bot.dxl.robot_name))
    time.sleep(move_time)

def move_grippers(bot_list, target_pose_list, move_time):
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t]
            bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(DT)

# setup_puppet_bot() is the shared entry point every RealEnv(setup_robots=True) caller goes
# through (run_vtla_policy.py, imitate_episodes_multi_gpu.py's real-robot eval, real_env.py's own
# test_real_teleop(), aloha_scripts/replay_episodes.py) before driving the puppet tick-by-tick via
# RealEnv.step(), which paces itself at DT internally (time.sleep(DT)) - so DT is the right
# default cadence here too, not some arbitrary "generic" duration. A caller that immediately does
# its own slower one-shot move (move_arms(), set_trajectory_time()) - e.g. run_vtla_policy.py's
# home_puppets(), or record_episodes_aloha2.py's opening_ceremony() via a *different* setup path
# (setup_robots=False, since it manages the arm's profile itself) - simply overwrites this right
# after, so defaulting to DT here costs nothing there and fixes it everywhere else.
DEFAULT_PUPPET_SETUP_MOVE_TIME = DT
DEFAULT_PUPPET_SETUP_ACCEL_TIME = min(DT, DT / 2)

def setup_puppet_bot(bot, moving_time=DEFAULT_PUPPET_SETUP_MOVE_TIME, accel_time=None):
    if accel_time is None:
        accel_time = min(DT, moving_time / 2)
    bot.dxl.robot_reboot_motors("single", "gripper", True)
    # profile_type defaults to "velocity" with profile_velocity=0 (i.e. no speed limit) when not
    # given explicitly - leaving the arm group in that state here meant every later move_arms()
    # call (which only ever touches the Profile_Velocity/Profile_Acceleration registers via
    # set_trajectory_time(), never the profile TYPE) had its moving_time misread by the servo as
    # a raw velocity/acceleration limit instead of a trajectory duration, so the puppet snapped to
    # its target near max speed regardless of the moving_time any caller asked for. Passing
    # profile_type="time" here - and syncing the python-side bookkeeping to match via
    # set_trajectory_time() - is what actually makes every later move_arms(..., move_time=...)
    # call behave the way its caller expects.
    bot.dxl.robot_set_operating_modes(
        "group", "arm", "position", "time",
        int(moving_time * 1000), int(accel_time * 1000),
    )
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    bot.arm.moving_time = None
    bot.arm.accel_time = None
    bot.arm.set_trajectory_time(moving_time=moving_time, accel_time=accel_time)
    torque_on(bot)

def setup_master_bot(bot):
    bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_off(bot)

def set_standard_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 800)
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)

def set_low_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 100)
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)

def torque_off(bot):
    bot.dxl.robot_torque_enable("group", "arm", False)
    bot.dxl.robot_torque_enable("single", "gripper", False)

def torque_on(bot):
    bot.dxl.robot_torque_enable("group", "arm", True)
    bot.dxl.robot_torque_enable("single", "gripper", True)
