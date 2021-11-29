# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import pickle
import argparse
import gym
import skvideo.io
import numpy as np
from pathlib import Path
from softlearning.environments.adapters.gym_adapter import GymAdapter
import time
import matplotlib.pyplot as plt

# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="Plots rollouts")

    parser.add_argument("-d", "--domain", type=str, help="domain name", default="SawyerDhandInHandPickUp")
    parser.add_argument("-e", "--task", type=str, help="task name", default="Fixed-v0")
    parser.add_argument("-p", "--policy",
                        type=str,
                        help="path to policy",
                        default="")
    parser.add_argument("-r", "--render",
                        type=str,
                        help="onscreen/offscreen rendering",
                        default="onscreen")
    parser.add_argument('-i', '--include',
                        type=str,
                        default='r3l',
                        help='task suite to import')
    parser.add_argument('-n', '--num_episodes',
                        type=int,
                        default=1,
                        help='number of episodes')
    parser.add_argument('-t', '--horizon_length',
                        type=int,
                        default=50,
                        help='rollout length')
    parser.add_argument('-f', '--filename',
                        type=str,
                        default='',
                        help='offline rendering video path')
    return parser.parse_args()

def main():
    # get args
    args = get_args()

    # load env
    if args.include is not "":
        exec("import " + args.include)


    env_params = {
        # 'init_xyz_range_params': {
        #     "type": "UniformRange",
        #     "values": [
        #         np.array([0.72, 0.15, 0.75]) - np.array([0.05, 0.05, 0]),
        #         np.array([0.72, 0.15, 0.75]) + np.array([0.05, 0.05, 0]),
        #     ],
        # },
        # 'random_init_angle': True,
        # 'reset_every_n_episodes': 1,
        # 'readjust_to_object_in_reset': True,
        # 'readjust_hand_xyz': True,
        # "readjust_hand_euler": False,
        'target_xyz_range_params': {
            'type': "DiscreteRange",
            "values": [
                np.array([0.92, 0.15, 1.2]),
            ],
        },
        'target_euler_range_params': {
            'type': "DiscreteRange",
            "values": [
                np.array([0, 0, 3 * np.pi / 2]),
            ],
        },
    }

    env = GymAdapter(
        domain=args.domain,
        task=args.task,
        **env_params,
    )

    if hasattr(env, "active_env"):
        obs_keys_to_log = env.active_env._observation_keys
        rew_keys_to_log = list(env.active_env._reward_keys_and_weights.keys())
    else:
        obs_keys_to_log = env.unwrapped._observation_keys + ("object_z_orientation", "target_z_orientation", "object_euler", "target_euler")
        rew_keys_to_log = list(env.unwrapped._reward_keys_and_weights.keys())

    rollout_imgs = []
    count_reward_imgs = []

    phased = hasattr(env, "num_phases")
    if phased:
        num_phases = env.num_phases
        phase_idx = 0
    for ep in range(args.num_episodes):
        env.reset()
        if phased:
            env.configure_phase(phase_idx)
            phase_idx = (phase_idx + 1) % num_phases
        ep_rewards = []
        for _ in range(args.horizon_length):
            obs, reward, done, info = env.step(env.action_space.sample())

            # Test policy going to its boundaries
            # act = np.zeros(env.action_space.shape)
            # act[-4] = 1
            # obs, reward, done, info = env.step(act)

            rollout_imgs.append(env.render(width=480, height=480, mode="rgb_array"))
            ep_rewards.append(reward)

            # count_rewards = env.get_obs_dict()["all_discretized_count_rewards"]
            # count_reward_imgs.append(count_rewards)

        obs_dict = env.get_obs_dict()
        rew_dict = env.get_reward_dict(None, obs_dict)
        print("\nObservations:")
        for key in obs_keys_to_log:
            if key in obs_dict:
                print(f"\t{key} = {obs_dict[key]}")
        print("\nRewards:")
        for key in rew_keys_to_log:
            if key in rew_dict:
                print(f"\t{key} = {rew_dict[key]}")

        ep_rewards = np.array(ep_rewards)
        print(f"\nEPISODE #{ep}")
        if len(ep_rewards) > 0:
            print(f"\tMean reward: {ep_rewards.mean()}")
            print(f"\tMax reward: {np.max(ep_rewards)}")
            print(f"\tMin reward: {np.min(ep_rewards)}")
            print(f"\tLast reward: {ep_rewards[-1]}")

    # for i, img in enumerate(count_reward_imgs):
    #     fig = plt.figure(figsize=(4, 4))
    #     plt.imshow(img)
    #     fig.savefig(f"reward_imgs/{i}.png")
    #     plt.close(fig)

    # import imageio
    # vid_imgs = []
    # for i in range(len(count_reward_imgs)):
    #     im = imageio.imread(f"reward_imgs/{i}.png")
    #     vid_imgs.append(im)
    # npskvideo.io.vwrite("reward_test.mp4", .asarray(vid_imgs))

    skvideo.io.vwrite(args.filename, np.asarray(rollout_imgs))
    print(f"Done saving videos to {args.filename}")

if __name__ == "__main__":
    main()
