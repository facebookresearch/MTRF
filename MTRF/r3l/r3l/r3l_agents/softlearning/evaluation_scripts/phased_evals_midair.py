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
import skvideo.io
from softlearning.environments.adapters.gym_adapter import GymAdapter

from softlearning.policies.utils import get_policy_from_variant
from softlearning.models.utils import flatten_input_structure
from softlearning.environments.utils import get_environment_from_params

from r3l.robot.default_configs import MOCAP_EULER_PALMDOWN
from r3l.utils.misc import save_video

def load_policy_from_checkpoint(ckpt_dir, env, phase_idx):
    # Load policy
    with open(os.path.join(ckpt_dir, "policy_params.pkl"), "rb") as f:
        policy_params = pickle.load(f)

    with open(os.path.join(ckpt_dir, "..", "params.pkl"), "rb") as f:
        variant = pickle.load(f)

    pickup_params = policy_params[phase_idx]

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

def do_evals(seed_dir, checkpoints_to_eval=None):
    print(seed_dir, "\n")
    path = Path(seed_dir)
    checkpoint_dirs = [d for d in glob.glob(str(path / "*")) if 'checkpoint' in d and os.path.isdir(d)]
    checkpoint_dirs.sort(key=lambda s: int(s.split("_")[-1]))
    if checkpoints_to_eval is not None:
        checkpoint_dirs = [d for d in checkpoint_dirs if int(d.split("_")[-1]) in checkpoints_to_eval]

    REPOS_MIDAIR_PHASE = 4
    N_EVAL_EPISODES = 20
    T = 100
    EVAL_T = 100
    EVAL_EVERY_N = 2
    MAX_ATTEMPTS = 6
    should_save_video = True

    env = None

    # env = GymAdapter(
    #     "SawyerDhandInHandValve3", "AllPhasesResetFree-v1",
    #     reset_every_n_episodes=np.inf,
    #     perturb_off=True,
    #     commanded_phase_changes=False,
    # )
    # env.reset()

    success_rates = []
    ckpt_numbers = []
    obs_dicts_per_policy = []
    rew_dicts_per_policy = []
    returns_per_policy = []
    for ckpt_dir in reversed(checkpoint_dirs[::EVAL_EVERY_N]):

        if env is None:
            with open(os.path.join(ckpt_dir, "..", "params.pkl"), "rb") as f:
                variant = pickle.load(f)

            env_params = variant["environment_params"]["training"]
            env_params["kwargs"]["commanded_phase_changes"] = False
            env = get_environment_from_params(env_params)
            # Create environment as softlearning expects it for policy initialization
            env = GymAdapter(None, None, env=env)
            # Never transition into perturbation for the reset controllers
            if hasattr(env, "turn_perturb_off"):
                env.turn_perturb_off()
            env.reset()

        env.unwrapped._envs[0].reset()

        ckpt_number = ckpt_dir.split("_")[-1]

        print("EVALUATING CHECKPOINT: ", ckpt_number)
        policies = [
            load_policy_from_checkpoint(ckpt_dir, env, phase)
            for phase in range(env.num_phases)
        ]

        successes = []
        obs_dicts = []
        rew_dicts = []
        returns = []
        frames = []

        for ep in range(N_EVAL_EPISODES):
            # Simplified run
            env.reset()
            phase = env.phase
            for t in range(T):
                env.step(policies[phase](env.get_obs_dict()))
                if should_save_video:
                    frames.append(env.render(mode="rgb_array", width=480, height=480))

            # env.reset()

            # phase = env.phase
            # # phase = env.update_phase()
            # if phase != REPOS_MIDAIR_PHASE:
            #     env.unwrapped._envs[1].reset() # Reset the reposition env to sample a random init position

            #     for _ in range(MAX_ATTEMPTS):
            #         # phase = env.update_phase()
            #         env.reset()
            #         phase = env.phase
            #         print("currently in phase:", phase)
            #         # We've reached the reposition midair task
            #         if phase == REPOS_MIDAIR_PHASE:
            #             break

            #         # env.active_env.reset_robot()
            #         for t in range(T):
            #             env.step(policies[phase](env.active_env.get_obs_dict()))
            #             if should_save_video:
            #                 frames.append(env.render(mode="rgb_array", width=480, height=480))

            #     env.set_goal(REPOS_MIDAIR_PHASE)

            # print("target xyz", env.get_obs_dict()["target_xyz"])
            # # Run the reposition midair evaluation episode

            # ret = 0
            # for t in range(EVAL_T):
            #     obs, rew, done, info = env.step(policies[REPOS_MIDAIR_PHASE](env.get_obs_dict()))
            #     if should_save_video:
            #         frames.append(env.render(mode="rgb_array", width=480, height=480))
            #     ret += rew

            # obs_dict = env.get_obs_dict()
            # rew_dict = env.get_reward_dict(None, obs_dict)
            # success = obs_dict["object_to_target_xyz_distance"] < 0.2
            # successes.append(success)
            # returns.append(ret)
            # obs_dicts.append(obs_dict)
            # rew_dicts.append(rew_dict)

        if should_save_video:
            save_video(f"./videos/{ckpt_number}.mp4", np.asarray(frames))

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
    # Valve
    # exp_path = Path("/home/justinvyu/ray_results_local/gym/SawyerDhandInHandValve3/AllPhasesResetFree-v1/2020-11-16T19-55-25-valve_phased_nomiddlepeg_randomxymidairrepos_noreorientphase")

    # Dodecahedron
    exp_path = Path("/home/justinvyu/ray_results/gym/SawyerDhandInHandDodecahedron/AllPhasesResetFree-v1/2020-12-01T01-45-49-dodec_full_phased")
    save_filename = "dodec_midair_phased_eval_data.pkl"

    seed_dirs = [d for d in glob.glob(str(exp_path / "*")) if os.path.isdir(d)]
    do_evals(seed_dirs[0], checkpoints_to_eval=[710])

    # from multiprocessing import Pool
    # with Pool(processes=len(seed_dirs)) as pool:
    #     eval_results = pool.map(do_evals, seed_dirs)

    # with(open(save_filename, "wb")) as f:
    #     pickle.dump(eval_results, f)
