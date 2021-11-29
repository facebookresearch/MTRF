# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import abc

class Robot(metaclass=abc.ABCMeta):
    def __init__(self, env=None):
        self._env = env
        if env:
            self._sim = env.sim
        else:
            self._sim = None

    @property
    @abc.abstractmethod
    def is_hardware(self):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abc.abstractmethod
    def set_state(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def get_obs_dict(self):
        raise NotImplementedError
