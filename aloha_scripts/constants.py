### Task parameters

# DATA_DIR = '/mnt/ssd1/guningquan/Programs_server/act_dataset_checkpoint/dataset'
# DATA_DIR = '/home/ubuntu20/guningquan/Programs_Codes/aloha_related/Dataset_and_Checkpoint/dataset'

import os
if os.getlogin() == 'guningquan':
    DATA_DIR = '/mnt/ssd2/guningquan/Dataset'
elif os.getlogin() == 'ubuntu20':
    DATA_DIR = '/home/robot/Dataset_and_Checkpoint/dataset'
else:
    raise ValueError(f"Unknown user: {os.getlogin()}")

TASK_CONFIGS = {

    'zip_tie_data_collection': {
        'dataset_dir': DATA_DIR + '/zip_tie_random',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'zip_tie': {
        'dataset_dir': DATA_DIR + '/zip_tie_random',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'zip_tie_normals': {
        'dataset_dir': DATA_DIR + '/zip_tie_random',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         # 'gel'
                         ]
    },

    'zip_tie_cotrain': {
        'dataset_dir': [
            DATA_DIR + '/zip_tie_random',
            DATA_DIR + '/mobile_aloha',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/zip_tie_random',
        ],
        'sample_weights': [7.5, 2.5],
        'train_ratio': 0.95,  # ratio of train data from the first dataset_dir
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },


    'aloha2_tactile_test': {
    'dataset_dir': DATA_DIR + '/aloha2_tactile_test',
    'episode_len': 1600,
    'camera_names': ['cam_high',
                        'cam_low',
                        'cam_left_wrist',
                        'cam_right_wrist',
                        ],
    'tactile_sensors': ['left_gripper_outer',
                        'left_gripper_inner',
                        'right_gripper_outer',
                        'right_gripper_inner',
                        ],
    'tactile_outputs': ['rectify',
                        'force',
                        ],
    # Which XENSE sensors to record. Any subset of the four the publisher exposes:
    #   left_gripper_inner, left_gripper_outer, right_gripper_inner, right_gripper_outer
    # Which fields to keep out of each sensor's /raw bundle. They must be a subset of what the
    # launch was started with, e.g.
    #   roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:="force,force_norm,difference"
    # ('rectify' is always present in the bundle, whatever extra_outputs says.)
    # Available: rectify, difference, depth, force, force_norm, force_resultant,
    #            marker2d, mesh3d, mesh3d_init, mesh3d_flow, timestamp
    #
    # Size matters here. Per sensor over a 1600-step episode, buffered in RAM during the
    # episode and then written to disk:
    #   force / force_norm / marker2d / mesh3d*   ~13 MB each   (35x20 fields)
    #   force_resultant / timestamp                negligible
    #   rectify / difference                      ~1.3 GB each  (700x400x3, JPEG'd on write)
    #   depth                                     ~1.8 GB       (700x400 float, not compressible)
    # The camera buffer alone is already ~5.9 GB per episode, so add the big ones deliberately.
    },

    'adapter_insertion': {
        'dataset_dir': DATA_DIR + '/adapter_insertion',
        'episode_len': 1600,
        'camera_names': ['cam_high',
                            'cam_low',
                            'cam_left_wrist',
                            'cam_right_wrist',
                            ],
        'tactile_sensors': ['left_gripper_outer',
                            'left_gripper_inner',
                            'right_gripper_outer',
                            'right_gripper_inner',
                            ],
        'tactile_outputs': ['rectify',
                            'force',
                            ],
        },


    'packaging': {
        'dataset_dir': DATA_DIR + '/packaging',
        'episode_len': 3000,
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },
    
    'packaging_move_box': {
        'dataset_dir': DATA_DIR + '/packaging_move_box',
        'episode_len': 1500,
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'packaging_simple': {
    'dataset_dir': DATA_DIR + '/packaging_simple',
    'episode_len': 2500,
    'camera_names': ['cam_high',
                        # 'cam_low',
                        'cam_left_wrist',
                        'cam_right_wrist',
                        ]
    },

    'packaging_simple_2': {
    'dataset_dir': DATA_DIR + '/packaging_simple_2',
    'episode_len': 2500,
    'camera_names': ['cam_high',
                        # 'cam_low',
                        'cam_left_wrist',
                        'cam_right_wrist',
                        ]
    },

    'packaging_simple_2_plus': {
    'dataset_dir': DATA_DIR + '/packaging_simple_2_plus',
    'episode_len': 2500,
    'camera_names': ['cam_high',
                        # 'cam_low',
                        'cam_left_wrist',
                        'cam_right_wrist',
                        ]
    },
}


### ALOHA fixed constants
DT = 0.02
FPS = round(1 / DT)  # derived from DT, not a separate literal - imitate_episodes_multi_gpu.py uses this as DT = 1/FPS

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
# Only the PUPPET pair is still live: it feeds the two PUPPET_GRIPPER_POSITION/VELOCITY
# helpers that real_env.get_qpos/get_qvel use. The MASTER pair has no remaining caller.
# MASTER_GRIPPER_POSITION_OPEN = 0.02417
# MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6]) - measured working range of the ALOHA2 grippers.
# The four grippers do not share one range, so every arm is calibrated separately.
# Re-measure with `python3 aloha_scripts/test_gripper_range.py <robot_name> --release`
# after any gripper is swapped or re-mounted.
MASTER_LEFT_GRIPPER_JOINT_OPEN = 0.7762
MASTER_LEFT_GRIPPER_JOINT_CLOSE = -0.0890
MASTER_RIGHT_GRIPPER_JOINT_OPEN = 0.8038
MASTER_RIGHT_GRIPPER_JOINT_CLOSE = -0.0522
PUPPET_LEFT_GRIPPER_JOINT_OPEN = 0.0767
PUPPET_LEFT_GRIPPER_JOINT_CLOSE = -1.5  # -1.3606
PUPPET_RIGHT_GRIPPER_JOINT_OPEN = 0.0782
PUPPET_RIGHT_GRIPPER_JOINT_CLOSE = -1.5  # -1.3898

# Backward-compatible single-sided aliases (they resolve to the RIGHT arm).
# Only the PUPPET pair still has importers: align.py and train_latent_model.py.
# MASTER_GRIPPER_JOINT_OPEN = MASTER_RIGHT_GRIPPER_JOINT_OPEN
# MASTER_GRIPPER_JOINT_CLOSE = MASTER_RIGHT_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_OPEN = PUPPET_RIGHT_GRIPPER_JOINT_OPEN
PUPPET_GRIPPER_JOINT_CLOSE = PUPPET_RIGHT_GRIPPER_JOINT_CLOSE

############################ Helper functions ############################
# Everything below that is commented out has no caller left anywhere in the repo.
# It is kept (not deleted) because it documents the original ALOHA gripper math.

# --- gripper position (metres) ---
# Used by real_env.get_qpos / get_qvel to normalise the reported finger position.
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

# MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
# MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
# PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
# MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))
# MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)


def _clamp01(x):
    """Clamp to [0, 1] so master over-travel cannot command the puppet past its limit."""
    return min(max(x, 0.0), 1.0)


# --- gripper joint (rad): the live teleop + deployment path ---
# master joint -> normalised action (real_env.get_action)
MASTER_LEFT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_LEFT_GRIPPER_JOINT_CLOSE) / (MASTER_LEFT_GRIPPER_JOINT_OPEN - MASTER_LEFT_GRIPPER_JOINT_CLOSE)
MASTER_RIGHT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_RIGHT_GRIPPER_JOINT_CLOSE) / (MASTER_RIGHT_GRIPPER_JOINT_OPEN - MASTER_RIGHT_GRIPPER_JOINT_CLOSE)
# normalised action -> puppet joint command (real_env.set_gripper_pose)
PUPPET_LEFT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_LEFT_GRIPPER_JOINT_OPEN - PUPPET_LEFT_GRIPPER_JOINT_CLOSE) + PUPPET_LEFT_GRIPPER_JOINT_CLOSE
PUPPET_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_RIGHT_GRIPPER_JOINT_OPEN - PUPPET_RIGHT_GRIPPER_JOINT_CLOSE) + PUPPET_RIGHT_GRIPPER_JOINT_CLOSE
# master joint -> puppet joint, clamped (one_side_teleop)
MASTER2PUPPET_LEFT_JOINT_FN = lambda x: PUPPET_LEFT_GRIPPER_JOINT_UNNORMALIZE_FN(_clamp01(MASTER_LEFT_GRIPPER_JOINT_NORMALIZE_FN(x)))
MASTER2PUPPET_RIGHT_JOINT_FN = lambda x: PUPPET_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN(_clamp01(MASTER_RIGHT_GRIPPER_JOINT_NORMALIZE_FN(x)))

# MASTER_GRIPPER_JOINT_NORMALIZE_FN = MASTER_RIGHT_GRIPPER_JOINT_NORMALIZE_FN
# PUPPET_LEFT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_LEFT_GRIPPER_JOINT_CLOSE) / (PUPPET_LEFT_GRIPPER_JOINT_OPEN - PUPPET_LEFT_GRIPPER_JOINT_CLOSE)
# PUPPET_RIGHT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_RIGHT_GRIPPER_JOINT_CLOSE) / (PUPPET_RIGHT_GRIPPER_JOINT_OPEN - PUPPET_RIGHT_GRIPPER_JOINT_CLOSE)
# PUPPET_GRIPPER_JOINT_NORMALIZE_FN = PUPPET_RIGHT_GRIPPER_JOINT_NORMALIZE_FN
# MASTER_LEFT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_LEFT_GRIPPER_JOINT_OPEN - MASTER_LEFT_GRIPPER_JOINT_CLOSE) + MASTER_LEFT_GRIPPER_JOINT_CLOSE
# MASTER_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_RIGHT_GRIPPER_JOINT_OPEN - MASTER_RIGHT_GRIPPER_JOINT_CLOSE) + MASTER_RIGHT_GRIPPER_JOINT_CLOSE
# MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = MASTER_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN
# PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = PUPPET_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN
# MASTER2PUPPET_JOINT_FN = MASTER2PUPPET_RIGHT_JOINT_FN

# --- position <-> joint conversions: no callers left ---
# MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
# MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
# PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
# PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

# --- master "half open" pose, used to home the grippers and to detect the start squeeze ---
MASTER_LEFT_GRIPPER_JOINT_MID = (MASTER_LEFT_GRIPPER_JOINT_OPEN + MASTER_LEFT_GRIPPER_JOINT_CLOSE) / 2
MASTER_RIGHT_GRIPPER_JOINT_MID = (MASTER_RIGHT_GRIPPER_JOINT_OPEN + MASTER_RIGHT_GRIPPER_JOINT_CLOSE) / 2
# MASTER_GRIPPER_JOINT_MID = MASTER_RIGHT_GRIPPER_JOINT_MID
