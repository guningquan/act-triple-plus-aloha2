from interbotix_xs_modules.arm import InterbotixManipulatorXS
try:
    from aloha_scripts.robot_utils import move_arms, torque_on, move_grippers, torque_off
    from aloha_scripts.constants import MASTER_LEFT_GRIPPER_JOINT_OPEN, MASTER_RIGHT_GRIPPER_JOINT_OPEN
    from aloha_scripts.constants import PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN
except ImportError:
    from robot_utils import move_arms, torque_on, move_grippers,torque_off
    from constants import MASTER_LEFT_GRIPPER_JOINT_OPEN, MASTER_RIGHT_GRIPPER_JOINT_OPEN
    from constants import PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN
import rospy
import threading,time
import argparse

# Order: (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate) - gripper is handled
# separately via move_grippers.
#
# OLD_*_SLEEP_POSITION are the values this file used to shut down to before this pass; they
# caused a hard drop when torque cut (most obviously master's wrist_angle=0.0). The factory
# "Sleep" pose from each arm model's own config (interbotix_xsarm_control/config/{vx300s,
# wx250s}.yaml: sleep_positions) was tried next, but is not accurate for this particular pair of
# arms (probably calibration/mounting/added-hardware dependent - the tactile sensors and gel
# pads on the puppet fingers alone shift the balance point the factory number assumes). So
# *_SLEEP_POSITION below is instead read directly off the live /puppet_*/joint_states and
# /master_*/joint_states while at rest with torque on, left/right averaged and the small
# (<0.02 rad) waist/forearm_roll/wrist_rotate components zeroed - those axes don't fight gravity
# in this configuration, unlike shoulder/elbow/wrist_angle, which are kept at the observed value.
# If it still drops on shutdown, the resting pose has likely drifted since this reading (re-run
# `rostopic echo /puppet_left/joint_states -n1` etc. and update these four tuples).
#
# Keep OLD_* only for the shutdown path's first way-point (see move_shut_robot): the current
# working pose during a session can be far from the sleep pose (large excursions on
# shoulder/elbow especially), and moving through the old, previously-used pose first keeps that
# initial large move on a path this arm has actually taken many times before the final,
# small correction onto the real rest pose, right before torque cuts.
OLD_PUPPET_SLEEP_POSITION = (0, -1.7, 1.55, 0.12, 0.65, 0)
OLD_MASTER_SLEEP_POSITION = (0, -1.76, 1.55, 0, 0.0, 0)

PUPPET_SLEEP_POSITION = (0, -1.85, 1.594, 0, 0.782, 0)   # measured resting pose, not the factory one
MASTER_SLEEP_POSITION = (0, -1.88, 1.605, 0, 0.281, 0)   # measured resting pose, not the factory one

# Multiplies every shutdown move_time below (gripper + both arm way-points). 1.0 = the original
# speed; 2.0 = twice as slow. move_arms()/move_grippers() take the target duration directly and
# do the interpolation themselves, so this alone controls how fast the arm travels - it does not
# need to be paired with any other change.
SHUTDOWN_SPEED_SCALE = 2.0

def move_robot(robot, gripper_position, arm_waypoints, move_times):
    """Like move_shut_robot, but leaves torque on afterwards (used for '--sleep': the arm should
    stay actively held at the rest pose, not powered down). Two stages by default for the same
    reason move_shut_robot uses them - see the module comment above: the old, well-traveled
    pose first, then the small final correction onto the measured rest pose, rather than one
    big jump straight from wherever the arm was working. Torque never leaves here, so - unlike
    move_shut_robot - there is no drop risk either way; this is purely about not swinging the
    arm through an untested straight-line path between two very different poses.
    """
    torque_on(robot)
    move_grippers([robot], [gripper_position], move_time=move_times[0])
    for arm_position, move_time in zip(arm_waypoints, move_times):
        move_arms([robot], [arm_position], move_time=move_time)

def move_shut_robot(robot, gripper_position, arm_waypoints, move_times):
    """Visit arm_waypoints (a sequence of 6-joint poses) in order, then cut torque.

    Two stages by default: the old sleep pose this file used to go straight to, then the arm
    model's real factory sleep pose (see the module-level comment above) - a small, final
    correction right before torque_off, rather than one big jump straight from whatever pose the
    arm happened to be commanded to before shutdown was requested.
    """
    torque_on(robot)
    move_grippers([robot], [gripper_position], move_time=move_times[0])
    for arm_position, move_time in zip(arm_waypoints, move_times):
        move_arms([robot], [arm_position], move_time=move_time)
    # move_arms() (robot_utils.py) now sets the servo's own moving_time to match and sleeps that
    # long itself, so by the time the loop above returns the arm really is at the last waypoint -
    # this is just a small settle margin, not compensating for a timing bug anymore.
    time.sleep(0.3)
    torque_off(robot)

def shut_down_all_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)

    puppet_waypoints = (OLD_PUPPET_SLEEP_POSITION, PUPPET_SLEEP_POSITION)
    master_waypoints = (OLD_MASTER_SLEEP_POSITION, MASTER_SLEEP_POSITION)
    puppet_move_times = tuple(t * SHUTDOWN_SPEED_SCALE for t in (1, 1))
    master_move_times = tuple(t * SHUTDOWN_SPEED_SCALE for t in (1, 1.5))

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_left, PUPPET_LEFT_GRIPPER_JOINT_OPEN, puppet_waypoints, puppet_move_times)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_right, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, puppet_waypoints, puppet_move_times)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(master_bot_left, MASTER_LEFT_GRIPPER_JOINT_OPEN, master_waypoints, master_move_times)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(master_bot_right, MASTER_RIGHT_GRIPPER_JOINT_OPEN, master_waypoints, master_move_times)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

def shut_down_puppet_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)

    puppet_waypoints = (OLD_PUPPET_SLEEP_POSITION, PUPPET_SLEEP_POSITION)
    puppet_move_times = tuple(t * SHUTDOWN_SPEED_SCALE for t in (1, 1))

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_left, PUPPET_LEFT_GRIPPER_JOINT_OPEN, puppet_waypoints, puppet_move_times)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_right, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, puppet_waypoints, puppet_move_times)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()


def sleep_all_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)

    puppet_waypoints = (OLD_PUPPET_SLEEP_POSITION, PUPPET_SLEEP_POSITION)
    master_waypoints = (OLD_MASTER_SLEEP_POSITION, MASTER_SLEEP_POSITION)
    puppet_move_times = (1, 1)
    master_move_times = (1, 1.5)

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_left, PUPPET_LEFT_GRIPPER_JOINT_OPEN, puppet_waypoints, puppet_move_times)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_right, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, puppet_waypoints, puppet_move_times)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_left, MASTER_LEFT_GRIPPER_JOINT_OPEN, master_waypoints, master_move_times)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_right, MASTER_RIGHT_GRIPPER_JOINT_OPEN, master_waypoints, master_move_times)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()


def sleep_puppet_robots():
    # rospy.init_node('sleep_plus')

    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)

    puppet_waypoints = (OLD_PUPPET_SLEEP_POSITION, PUPPET_SLEEP_POSITION)
    puppet_move_times = (1, 1)

    threads = []
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_left, PUPPET_LEFT_GRIPPER_JOINT_OPEN, puppet_waypoints, puppet_move_times)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_right, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, puppet_waypoints, puppet_move_times)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

def sleep_master_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)

    master_waypoints = (OLD_MASTER_SLEEP_POSITION, MASTER_SLEEP_POSITION)
    master_move_times = (1, 1.5)

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_left, MASTER_LEFT_GRIPPER_JOINT_OPEN, master_waypoints, master_move_times)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_right, MASTER_RIGHT_GRIPPER_JOINT_OPEN, master_waypoints, master_move_times)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()





def main():
    parser = argparse.ArgumentParser(description="Control script for robots")
    parser.add_argument("--sleep", action="store_true",
                        help="Put all robots into sleep mode")
    parser.add_argument("--shut_down", action="store_true",
                        help="Shut down all robots")
    parser.add_argument("--shut_down_puppet", action="store_true",
                        help="Shut down puppet robots")

    args = parser.parse_args()

    if args.sleep:
        sleep_all_robots()
    elif args.shut_down:
        shut_down_all_robots()
    elif args.shut_down_puppet:
        shut_down_puppet_robots()
    else:
        print("Please provide a valid argument (--sleep or --shut_down).")

if __name__ == "__main__":
    rospy.init_node('sleep_plus')
    main()