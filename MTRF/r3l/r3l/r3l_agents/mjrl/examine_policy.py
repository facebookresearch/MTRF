# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

from os import environ
environ["MKL_THREADING_LAYER"] = "GNU"

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
import pickle
import argparse

# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="Plots rollouts")

    parser.add_argument("-e", "--env_name",
                        type=str,
                        help="environment name",
                        default="SawyerDhandInHandPickup-v3")
    parser.add_argument("-p", "--policy",
                        type=str,
                        help="path to policy",
                        default="")
    parser.add_argument("-r", "--render",
                        type=str,
                        help="onscreen/offscree rendering",
                        default="onscreen")
    parser.add_argument('-i', '--include',
                        type=str,
                        default='r3l',
                        help='task suite to import')
    parser.add_argument('-n', '--num_episodes',
                        type=int,
                        default=10,
                        help='number of episodes')
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
    e = GymEnv(args.env_name)

    # load policy
    policy = args.policy
    if args.policy == "":
        pol = MLP(e.spec, init_log_std=0.1)
        mode = "exploration"
    else:
        pol = pickle.load(open(policy, 'rb'))
        mode = "evaluation"

    # Visualized policy
    if args.render == "onscreen":
        # On screen
        e.env.env.visualize_policy(
            pol,
            horizon=e.horizon,
            num_episodes=args.num_episodes,
            mode=mode)
    else:
        # Offscreen buffer
        e.env.env.visualize_policy_offscreen(
            pol,
            horizon=100,
            num_episodes=args.num_episodes,
            mode=mode,
            filename=args.filename)

    # Close envs
    e.env.env.close_env()

if __name__ == '__main__':
    main()
