# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import abc
import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
from r3l.utils.mocap_utils import *
from gym.spaces import Box

class SawyerDhandBaseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, model_path, frame_skip=40,
        mocap_range=(.01, .01, .01, .25, .25, .25),
        mocap_pos_mean=None,
        mocap_quat_mean=None,
        mocap_velocity_lim=None,
                 **kwargs):
        utils.EzPickle.__init__(self)
        self.initializing = True

        self.target = 0
        self.grasp = 0
        mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip)
        self.target = self.sim.model.site_name2id('target')
        self.grasp = self.sim.model.site_name2id('grasp')

        self.wrist_act = mocap_act(self, "mocap", np.array(mocap_range),
                                   mocap_pos_mean=mocap_pos_mean, mocap_quat_mean=mocap_quat_mean, velocity_lim=mocap_velocity_lim)

        # TODO: Need to change this to bias things
        self.act_mid = 0.5*self.model.actuator_ctrlrange[:, 0] + 0.5*self.model.actuator_ctrlrange[:, 1]
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.initializing = False
        self.action_space = Box(low=-1*np.ones_like((self.action_space.low)),
                                high=np.ones_like((self.action_space.high)))

    def step(self, a):
        a_end = a[16:]
        a = a[:16]
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
            print("WARNING: Actions cannot be remapped. (Expected during initialization)")

        # Avoiding initialization
        if len(a_end) > 0:
            self.wrist_act.update(self, a_end)

        self.do_simulation(a, self.frame_skip)
        obs = self.get_obs()

        score, reward_dict, solved, done = self._get_score_reward_solved_done(self.obs_dict)

        # finalize step
        env_info = {
            'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
            'reward': reward_dict,
            'score': score,
            'solved': solved
        }
        self.viewer_render()
        return obs, reward_dict['total'], done, env_info

    def get_obs(self):
        raise NotImplementedError

    def _get_score_reward_solved_done(self):
        raise NotImplementedError

    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs = paths["observations"]
        score, rewards, done = self._get_score_reward_solved_done(obs)
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()

    @abc.abstractmethod
    def reset_model(self):
        raise NotImplementedError

    def evaluate_success(self, paths, logger=None):
        success = 0.0
        for p in paths:
            if np.mean(p['env_infos']['solved'][-4:]) > 0.0:
                success += 1.0
        success_rate = 100.0*success/len(paths)
        if logger is None:
            # nowhere to log so return the value
            return success_rate
        else:
            # log the success
            # can log multiple statistics here if needed
            logger.log_kv('success_rate', success_rate)
            return None

    # --------------------------------
    # get and set states
    # --------------------------------
    def get_env_state(self):
        return dict(qp=self.data.qpos.copy(),
                    qv=self.data.qvel.copy(),
                    mocap_pos=self.data.mocap_pos.copy(),
                    mocap_quat=self.data.mocap_quat.copy(),
                    site_pos=self.data.site_xpos.copy(),
                    body_pos=self.model.body_pos.copy(),
                    body_quat=self.model.body_quat.copy(),
                    )

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        self.set_state(qp, qv)
        self.sim.data.mocap_pos[:] = state['mocap_pos']
        self.sim.data.mocap_quat[:] = state['mocap_quat']
        self.sim.data.site_xpos[:] = state['site_pos']
        self.sim.model.body_pos[:] = state['body_pos']
        self.sim.model.body_quat[:] = state['body_quat']
        self.sim.forward()

    # --------------------------------
    # utility functions
    # --------------------------------
    def get_env_infos(self):
        return dict(state=self.get_env_state())

    # --------------------------------
    # rendering functions
    # --------------------------------
    def render(self, mode="rgb_array", **kwargs):
        if mode == "rgb_array" or "offscreen":
            return self.sim.render(mode="offscreen", **kwargs)
        elif mode == "human" or "onscreen":
            self.viewer_render()
        else:
            raise ValueError(f"Invalid mode = {mode}")

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = -90
        self.viewer.cam.distance = 2.5
        self.viewer.cam.elevation = -30

        self.sim.forward()

    def viewer_render(self):
        if not hasattr(self, 'viewer') or self.viewer is None:
            self.mj_viewer_setup()
        self.viewer.render()

    def close_env(self):
        pass