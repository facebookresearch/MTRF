# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import numpy as np

def match_hand_xyz_euler(hand_xyz, hand_euler, env, camera_env=None, T=100, save_frames=False, image_size=(512, 512)):
    t = 0
    frames = []
    if camera_env is None:
        camera_env = env
    while (t < T
            and (np.linalg.norm(env.get_obs_dict()["mocap_pos"] - hand_xyz) > 0.03
            or np.linalg.norm(env.get_obs_dict()["mocap_euler"] - hand_euler) > 0.2)):
        a = np.zeros(22)
        xyz_diff = hand_xyz - env.get_obs_dict()["mocap_pos"]
        xyz_diff[np.abs(xyz_diff) < 0.01] = 0
        if np.linalg.norm(xyz_diff) < 0.05:
            xyz_act = np.sign(xyz_diff) / 10
        else:
            xyz_act = np.sign(xyz_diff) / 2

        euler_diff = hand_euler - env.get_obs_dict()["mocap_euler"]
        euler_diff[np.abs(euler_diff) < 0.05] = 0
        if np.linalg.norm(euler_diff) < 0.25:
            euler_act = np.sign(euler_diff) / 2
        else:
            euler_act = np.sign(euler_diff)

        a[-6:] = np.concatenate([xyz_act, euler_act])
        env.step(a)
        env.do_simulation(np.zeros(16), 40)
        if save_frames:
            frames.append(camera_env.render(mode="rgb_array", width=image_size[0], height=image_size[1]))
        t += 1
    return frames