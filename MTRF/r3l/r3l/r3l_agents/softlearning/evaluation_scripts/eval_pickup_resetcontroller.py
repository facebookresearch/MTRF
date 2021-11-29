# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import numpy as np
from pathlib import Path
import glob
import pickle
import matplotlib.pyplot as plt

import os
import r3l
import gym
from r3l.r3l_envs.inhand_env.pickup import SawyerDhandInHandObjectPickupFixed
from softlearning.environments.adapters.gym_adapter import GymAdapter

exp_path = Path("/home/justinvyu/shared/ray_results/gym/SawyerDhandInHandValve3/RepositionPickupResetFree-v0/2020-10-30T04-41-28-valve_pickup_reset_controller_baseline_2phases/")

env = GymAdapter("SawyerDhandInHandValve3", "PickupFixed-v0")
env.reset()

seed_dirs = [d for d in glob.glob(str(exp_path / "*")) if os.path.isdir(d)]
for seed_dir in seed_dirs:
    checkpoint_dirs = [
        d for d in glob.glob(str(Path(seed_dir) / "*"))
        if 'checkpoint' in d and os.path.isdir(d)
    ]
    checkpoint_dirs.sort(key=lambda s: int(s.split("_")[-1]))

