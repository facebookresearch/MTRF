# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import numpy as np
MOCAP_POS_PALMDOWN = np.array([0.705, 0.185, 1.0])
MOCAP_QUAT_PALMDOWN = np.array([0, 1, 0, 0])
MOCAP_EULER_PALMDOWN = np.array([-np.pi, 0, 0])
ARM_QPOS_PALMDOWN = np.array([0.7,  0.133, -0.503, 1.067, -2.308, 0.976,  0.0973])

MOCAP_POS_PALMUP = np.array([0.705, 0.185, 1.0])
MOCAP_QUAT_PALMUP = np.array([1, 0, 0, 0])
MOCAP_EULER_PALMUP = np.array([0, 0, 0])
ARM_QPOS_PALMUP = np.array([0.758,  0.403, -0.953,  0.935, -2.019, 0.774, -2.811])

DEFAULT_MOCAP_RANGE = np.array([.01, .01, .01, .25, .25, .25])
DEFAULT_MOCAP_VELOCITY_LIM = 0.01 * np.ones((6, ))

WRIST_TO_HAND_X_OFFSET = 0.135

# Dhand Configs

DEFAULT_DHAND_ROBOT_CONFIG = {
    "qpos_indices": list(range(7, 7 + 16)),
    "qvel_indices": list(range(7, 7 + 16)),
    "qpos_range": None,
    "qvel_range": None,
    "control_mode": "deltapos",
}

PI = 3.1415
# Reference calibration map
DEFAULT_DHAND_CALIBRATION_MAP = {
    # Finger 1
    10: (1, -PI),
    11: (1, -3 * PI / 2),
    12: (1, -PI),
    13: (1, -PI),
    # Finger 2
    20: (1, -PI),
    21: (1, -3 * PI / 2),
    22: (1, -PI),
    23: (1, -PI),
    # Finger 3
    30: (1, -PI),
    31: (1, -3 * PI / 2),
    32: (1, -PI),
    33: (1, -PI),
    # Thumb
    # TODO: figure out correct calibrations for these
    40: (1, -220 * PI / 180),
    41: (1, -PI),
    42: (1, -PI),
    43: (1, -PI),
}

HARDWARE_DEFAULT_DHAND_ROBOT_CONFIG = {
    "qpos_indices": list(range(7, 7 + 16)),
    "qvel_indices": list(range(7, 7 + 16)),
    "qpos_range": np.array([
        (-0., 0.),
        (-0.61 * np.pi, 0.027 * np.pi),
        (-np.pi / 2, np.pi / 2),
        (-np.pi / 2, 0)
    ] * 3 + [
        (-100 * np.pi / 180, 0.0),(-np.pi / 2, np.pi / 2),(-np.pi / 2, np.pi / 2),(-np.pi / 2, 0)]),
    "qvel_range": None,
    "calib_offset": np.array([
        -PI, -3 * PI / 2, -PI, -PI,         # 10, 11, 12, 13
        -PI, -3 * PI / 2, -PI, -PI,         # 20, 21, 22, 23
        -PI, -3 * PI / 2, -PI, -PI,         # 30, 31, 32, 33
        220 * PI / 180, PI, -PI, -PI,     # 40, 41, 42, 43
    ]),
    "calib_scale": np.array([
        1, 1, 1, 1,     # 10, 11, 12, 13
        1, 1, 1, 1,     # 20, 21, 22, 23
        1, 1, 1, 1,     # 30, 31, 32, 33
        -1, -1, 1, 1,     # 40, 41, 42, 43
    ]),
    "device_path": "/dev/ttyUSB0",
    "motor_ids": [
        10, 11, 12, 13,
        20, 21, 22, 23,
        30, 31, 32, 33,
        40, 41, 42, 43,
    ],
}

HARDWARE_REACHING_DHAND_ROBOT_CONFIG = {
    "qpos_indices": list(range(7, 7 + 16)),
    "qvel_indices": list(range(7, 7 + 16)),
    "qpos_range": np.array([
        (-0., 0.),
        (-0.61 * np.pi, 0.027 * np.pi),
        (-np.pi / 2, np.pi / 2),
        (-np.pi / 2, 0)
    ] * 3 + [
        (-100 * np.pi / 180, 0.0),(-np.pi / 2, np.pi / 2),(-np.pi / 2, np.pi / 2),(-np.pi / 2, 0)]),
    "qvel_range": None,
    "calib_offset": np.array([
        -PI, -3 * PI / 2, -PI, -PI,         # 10, 11, 12, 13
        -PI, -3 * PI / 2, -PI, -PI,         # 20, 21, 22, 23
        -PI, -3 * PI / 2, -PI, -PI,         # 30, 31, 32, 33
        220 * PI / 180, PI, -PI, -PI,     # 40, 41, 42, 43
    ]),
    "calib_scale": np.array([
        1, 1, 1, 1,     # 10, 11, 12, 13
        1, 1, 1, 1,     # 20, 21, 22, 23
        1, 1, 1, 1,     # 30, 31, 32, 33
        -1, -1, 1, 1,     # 40, 41, 42, 43
    ]),
    "device_path": "/dev/ttyUSB0",
    "motor_ids": [
        10, 11, 12, 13,
        20, 21, 22, 23,
        30, 31, 32, 33,
        40, 41, 42, 43,
    ],
    "hand_velocity_lim": np.zeros(16),
}

# =========== SIM SAWYER CONFIGS =========== #

DEFAULT_SAWYER_ROBOT_CONFIG = {
    "qpos_indices": list(range(7)),
    "qvel_indices": list(range(7)),
    "qpos_range": None,
    "qvel_range": None,
    "mocap_name": "mocap",
    "mocap_range": (
        (0.705 - 0.3, 0.705 + 0.3),
        (0.185 - 0.3, 0.185 + 0.3),
        (1.0 - 0.25, 1.0 + 0.25),
        (-np.pi, -np.pi),
        (0, 0),
        (0 - np.pi / 6, 0 + np.pi / 6),
    ),
    "mocap_velocity_lim": np.array([0.01, 0.01, 0.01, 0, 0, 0.02]),
    "control_mode": "deltapos",
}

REPOSITION_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
REPOSITION_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        (0.705 - 0.45, 0.705 + 0.4),
        (0.185 - 0.4, 0.185 + 0.4),
        (1.0 - 0.1, 1.0 + 0.1),
        (-np.pi, -np.pi),
        (0, 0),
        (0 - np.pi / 3, 0 + np.pi / 3),
    ),
})
# REPOSITION_SAWYER_ROBOT_CONFIG.update({
#     "mocap_range": (
#         (0.705 - 0.3, 0.705 + 0.3),
#         (0.185 - 0.3, 0.185 + 0.3),
#         (1.0 - 0.1, 1.0 + 0.1),
#         (-np.pi, -np.pi),
#         (0, 0),
#         (0 - np.pi / 4, 0 + np.pi / 4),
#     ),
# })

REORIENT_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
REORIENT_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        (0.705 - 0.3 - 0.135, 0.705 + 0.4 - 0.135),
        (0.185 - 0.4, 0.185 + 0.4),
        (1.0 - 0.1, 1.0 + 0.1),
        (-np.pi, -np.pi),
        (0, 0),
        (0 - np.pi / 3, 0 + np.pi / 3),
    ),
    # "mocap_range": (
    #     (0.705 - 0.15, 0.705 + 0.15),
    #     (0.185 - 0.15, 0.185 + 0.15),
    #     (1.0 - 0.075, 1.0 + 0.075),
    #     (-np.pi, -np.pi),
    #     (0, 0),
    #     (0 - np.pi / 3, 0 + np.pi / 3),
    # ),
})

PICKUP_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
PICKUP_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        # 0.135 is the wrist to hand offset
        (0.705 - 0.15 - 0.135, 0.705 + 0.15 - 0.135),
        (0.185 - 0.15, 0.185 + 0.15),
        (1.0 - 0.175, 1.0 + 0.1),
        (-np.pi, -np.pi),
        (0, 0),
        (0 - np.pi / 6, 0 + np.pi / 6),
    ),
})

FLIPDOWN_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
FLIPDOWN_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        (MOCAP_POS_PALMUP[0] - 0.15, MOCAP_POS_PALMUP[0] + 0.15),
        (MOCAP_POS_PALMUP[1] - 0.15, MOCAP_POS_PALMUP[1] + 0.15),
        (MOCAP_POS_PALMUP[2] - 0.175, MOCAP_POS_PALMUP[2] + 0.175),
        (MOCAP_EULER_PALMUP[0] - np.pi, MOCAP_EULER_PALMUP[0] + np.pi),
        (MOCAP_EULER_PALMUP[1], MOCAP_EULER_PALMUP[1]),
        (MOCAP_EULER_PALMUP[2] - np.pi / 6, MOCAP_EULER_PALMUP[2] + np.pi / 6)
    ),
    "mocap_velocity_lim": np.array([0, 0, 0, 0.1, 0, 0])
})

FLIPUP_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
FLIPUP_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        (MOCAP_POS_PALMDOWN[0] - 0.3, MOCAP_POS_PALMDOWN[0] + 0.3),
        (MOCAP_POS_PALMDOWN[1] - 0.3, MOCAP_POS_PALMDOWN[1] + 0.3),
        # NOTE: This matches the pickup range so that there is no clipping on
        # phase transition
        (MOCAP_POS_PALMDOWN[2] - 0.175, MOCAP_POS_PALMDOWN[2] + 0.1),
        (MOCAP_EULER_PALMDOWN[0] - np.pi, MOCAP_EULER_PALMDOWN[0] + np.pi),
        (MOCAP_EULER_PALMDOWN[1], MOCAP_EULER_PALMDOWN[1]),
        (MOCAP_EULER_PALMDOWN[2] - np.pi / 6, MOCAP_EULER_PALMDOWN[2] + np.pi / 6),
    ),
    # NOTE: The 0.01 allowed velocity in Z is to allow the Sawyer to settle
    # back to the target xyz position after having flipped up.
    "mocap_velocity_lim": np.array([0, 0, 0.01, 0.1, 0, 0.05])
})

MIDAIR_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
MIDAIR_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        (MOCAP_POS_PALMUP[0] - 0.3, MOCAP_POS_PALMUP[0] + 0.3),
        (MOCAP_POS_PALMUP[1] - 0.3, MOCAP_POS_PALMUP[1] + 0.3),
        (MOCAP_POS_PALMUP[2] - 0.1, MOCAP_POS_PALMUP[2] + 0.1),
        (MOCAP_EULER_PALMUP[0] - np.pi / 6, MOCAP_EULER_PALMUP[0] + np.pi / 6),
        (MOCAP_EULER_PALMUP[1], MOCAP_EULER_PALMUP[1]),
        (MOCAP_EULER_PALMUP[2] - np.pi / 6, MOCAP_EULER_PALMUP[2] + np.pi / 6),
    ),
    "mocap_velocity_lim": np.array([0.0025, 0.0025, 0.005, 0.01, 0, 0.01])
})

MIDAIR_RESETFREE_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
MIDAIR_RESETFREE_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        # Ranges match the max of all the other phase's ranges
        (MOCAP_POS_PALMUP[0] - 0.45, MOCAP_POS_PALMUP[0] + 0.4),
        (MOCAP_POS_PALMUP[1] - 0.4, MOCAP_POS_PALMUP[1] + 0.4),
        (MOCAP_POS_PALMUP[2] - 0.175, MOCAP_POS_PALMUP[2] + 0.1),
        (MOCAP_EULER_PALMUP[0] - np.pi, MOCAP_EULER_PALMUP[0] + np.pi),
        (MOCAP_EULER_PALMUP[1], MOCAP_EULER_PALMUP[1]),
        (MOCAP_EULER_PALMUP[2] - np.pi / 6, MOCAP_EULER_PALMUP[2] + np.pi / 6),
    ),
    "mocap_velocity_lim": np.array([0.01, 0.01, 0.01, 0.1, 0, 0.1])
})

PICKUP_FLIPUP_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
PICKUP_FLIPUP_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        # Ranges match the max of all the other phase's ranges
        (MOCAP_POS_PALMDOWN[0] - 0.45, MOCAP_POS_PALMDOWN[0] + 0.4),
        (MOCAP_POS_PALMDOWN[1] - 0.4, MOCAP_POS_PALMDOWN[1] + 0.4),
        (MOCAP_POS_PALMDOWN[2] - 0.175, MOCAP_POS_PALMDOWN[2] + 0.1),
        (MOCAP_EULER_PALMDOWN[0] - np.pi, MOCAP_EULER_PALMDOWN[0] + np.pi),
        (MOCAP_EULER_PALMDOWN[1], MOCAP_EULER_PALMDOWN[1]),
        (MOCAP_EULER_PALMDOWN[2] - np.pi / 6, MOCAP_EULER_PALMDOWN[2] + np.pi / 6),
    ),
    "mocap_velocity_lim": np.array([0.01, 0.01, 0.01, 0.1, 0, 0.1])
})

POSE_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
POSE_SAWYER_ROBOT_CONFIG.update({
    # Allow all kinds of configurations, but don't let the Sawyer move
    "mocap_range": (
        # 0.135 is the wrist to hand offset
        (0.705 - 0.25 - 0.135, 0.705 + 0.25 - 0.135), # middle +/- 0.25
        (0.185 - 0.25, 0.185 + 0.25), # middle +/- 0.25
        (1.0 - 0.175, 1.0 + 0.3),
        (-np.pi - np.pi, -np.pi + np.pi),
        (0 - np.pi, 0 + np.pi),
        (0 - np.pi, 0 + np.pi),
    ),
    "mocap_velocity_lim": np.array([0, 0, 0, 0, 0, 0]),
})

# =========== HARDWARE SAWYER CONFIGS =========== #
HARDWARE_DEFAULT_SAWYER_ROBOT_CONFIG = {
    "qpos_indices": list(range(7)),
    "qvel_indices": list(range(7)),
    "qpos_range": None,
    "qvel_range": None,
    "mocap_name": "mocap",
    "mocap_range": (
        (0.71, 1.0),
        (-0.1, 0.25),
        (0.08, 0.3), # 0.08
        # (0.3, 0.3), # was (0.15, 0.3),
        (0, 0),  # ROLL
        (np.pi / 2, np.pi / 2),  # YAW
        (np.pi / 2, np.pi / 2)       # PITCH
    ),
    "mocap_velocity_lim": np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01]),
    "control_mode": "deltapos",
}

HARDWARE_POSING_SAWYER_ROBOT_CONFIG = {
    # "qpos_indices": list(range(7)),
    # "qvel_indices": list(range(7)),
    # "qpos_range": None,
    # "qvel_range": None,
    # "mocap_name": "mocap",
    # "mocap_range": (
    #     (0.475, 0.475),
    #     (-0.08, -0.08),
    #     (0.2375, 0.2375),
    #     (0, 0),  # ROLL
    #     (np.pi / 2, np.pi / 2),  # YAW
    #     (np.pi / 2, np.pi / 2)       # PITCH
    # ),
    # "mocap_velocity_lim": np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01]),
    # "control_mode": "deltapos",
}

# Object Configs

DEFAULT_OBJECT_CONFIG = {
    "qpos_indices": list(range(7 + 16, 7 + 16 + 7)),
    "qvel_indices": list(range(7 + 16, 7 + 16 + 6)),
    "qpos_range": None,
    "qvel_range": None,
}

HARDWARE_DEFAULT_OBJECT_CONFIG = {
    "qpos_indices": list(range(7 + 16, 7 + 16 + 7)),
    "qvel_indices": list(range(7 + 16, 7 + 16 + 6)),
    "qpos_range": None,
    "qvel_range": None,
    "calib_offset": np.zeros(7),
    "calib_scale": np.zeros(7),
    "device_path": "/dev/ttyUSB2",
    # "motor_ids": [10, 20],
    "motor_ids": [10,],
}

