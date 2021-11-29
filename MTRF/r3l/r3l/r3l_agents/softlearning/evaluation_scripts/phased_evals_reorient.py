# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import sys
import numpy as np
from pathlib import Path
import glob
import pickle
import matplotlib.pyplot as plt
import os
import r3l
import gym
import imageio
from r3l.r3l_envs.inhand_env.pickup import SawyerDhandInHandObjectPickupFixed
from softlearning.environments.adapters.gym_adapter import GymAdapter

from softlearning.policies.utils import get_policy_from_variant
from softlearning.models.utils import flatten_input_structure

def save_video(filename, video_frames, fps=60):
    assert fps == int(fps), fps
    import skvideo.io
    skvideo.io.vwrite(filename, video_frames, inputdict={'-r': str(int(fps))})

def load_policy_from_checkpoint(ckpt_dir, env):
    # Load policy
    with open(os.path.join(ckpt_dir, "policy_params.pkl"), "rb") as f:
        policy_params = pickle.load(f)

    with open(os.path.join(ckpt_dir, "..", "params.pkl"), "rb") as f:
        variant = pickle.load(f)

    pickup_params = policy_params[1] # <- reorient params = index 1

    policy = get_policy_from_variant(variant, env)
    policy.set_weights(pickup_params)
    return wrap_policy(policy)

def wrap_policy(policy):
    def wrapped_policy(obs_dict):
        feed_dict = {
            key: obs_dict[key][None, ...]
            for key in policy.observation_keys
        }
        observation = flatten_input_structure(feed_dict)
        with policy.set_deterministic(True):
            action = policy.actions_np(observation)[0]
        return action
    return wrapped_policy

def do_evals(seed_dir):
    print(seed_dir, "\n")
    path = Path(seed_dir)
    checkpoint_dirs = [d for d in glob.glob(str(path / "*")) if 'checkpoint' in d and os.path.isdir(d)]
    checkpoint_dirs.sort(key=lambda s: int(s.split("_")[-1]), reverse=True)

    N_EVAL_EPISODES = 1
    T = 50
    EVAL_EVERY_N = 2

    env = GymAdapter(
        "SawyerDhandInHandValve3",
        "ReorientFixed-v0",
        init_xyz_range_params={
            "type": "UniformRange",
            "values": [np.array([0.72, 0.15, 0.78]), np.array([0.72, 0.15, 0.78])],
        },
        init_euler_range_params={
            "type": "UniformRange",
            "values": [np.array([0, 0, -np.pi]), np.array([0, 0, np.pi])],
        },
        reset_every_n_episodes=1,
        reset_robot=True,
        readjust_to_object_in_reset=True,
    )
    env.reset()

    success_rates = []
    ckpt_numbers = []
    obs_dicts_per_policy = []
    rew_dicts_per_policy = []
    returns_per_policy = []

    for ckpt_dir in checkpoint_dirs[::EVAL_EVERY_N]:
        ckpt_number = ckpt_dir.split("_")[-1]
        print("EVALUATING CHECKPOINT: ", ckpt_number)
        policy = load_policy_from_checkpoint(ckpt_dir, env)

        successes = []
        obs_dicts = []
        rew_dicts = []
        returns = []
        frames = []
        for ep in range(N_EVAL_EPISODES):
            env.reset()
            while env.get_obs_dict()["object_to_target_mod_120_circle_distance"] < 0.2:
                env.reset()
            ret = 0
            for t in range(T):
                _, rew, done, info = env.step(policy(env.get_obs_dict()))
                ret += rew
                frames.append(env.render(mode="rgb_array", width=480, height=480))

            obs_dict = env.get_obs_dict()
            rew_dict = env.get_reward_dict(None, obs_dict)
            print(obs_dict["object_to_target_mod_120_circle_distance"])
            success = obs_dict["object_to_target_mod_120_circle_distance"] < 0.2
            successes.append(success)
            returns.append(ret)
            obs_dicts.append(obs_dict)
            rew_dicts.append(rew_dict)

        video_name = f"./videos/reorient/phased_checkpoint_{ckpt_number}.mp4"
        save_video(video_name, np.asarray(frames), fps=40)

        ckpt_numbers.append(ckpt_number)
        success_rate = np.array(successes).astype(int).mean()
        print("success % = ", success_rate)
        success_rates.append(success_rate)
        obs_dicts_per_policy.append(obs_dicts)
        rew_dicts_per_policy.append(rew_dicts)
        returns_per_policy.append(np.mean(returns))
        break

    return {
        "iters": ckpt_numbers,
        "success": success_rates,
        "obs": obs_dicts_per_policy,
        "rew": rew_dicts_per_policy,
        "returns": returns_per_policy,
    }

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-s", "--save_filename", help="save filename", type=str)
    # parser.add_argument("-p", "--exp_path", help="top level experiment path", type=str)

    exp_path = Path("/home/justinvyu/ray_results/gym/SawyerDhandInHandValve3/RepositionReorientPickupPerturbResetFree-v0/2020-10-28T19-43-42-4phase_fixedsawyerxzrange_newobskeys_repos_to_middle")
    # save_filename = "reorient_phased_eval_data.pkl"

    # seed_dirs = [d for d in glob.glob(str(exp_path / "*")) if os.path.isdir(d)]
    # # do_evals(seed_dirs[0])
    # from multiprocessing import Pool
    # with Pool(processes=len(seed_dirs)) as pool:
    #     eval_results = pool.map(do_evals, seed_dirs)

    # with(open(save_filename, "wb")) as f:
    #     pickle.dump(eval_results, f)

    seed_dirs = [d for d in glob.glob(str(exp_path / "*")) if os.path.isdir(d)]
    seed_dirs = [d for d in seed_dirs if "seed=470" in d]
    print(seed_dirs)
    do_evals(seed_dirs[0])
