# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import abc
from typing import Sequence, Optional
import numpy as np


class Range(metaclass=abc.ABCMeta):
    def __init__(self, np_random, values):
        self.np_random = np_random
        self.values = values

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError


class DiscreteRange(Range):
    def __init__(
            self,
            np_random,
            values: Sequence[np.ndarray],
            start_idx: int = 0,
            choosing_strategy: str = "random",
            probs: Optional[np.ndarray] = None,
    ):
        super(DiscreteRange, self).__init__(np_random, values)
        self._idx = start_idx
        assert choosing_strategy in ("random", "cycle")
        self._choosing_strategy = choosing_strategy
        self._probs = probs
        if probs is None:
            self._probs = np.ones(len(values)) / len(values)
        else:
            assert len(probs) == len(values)

    def __len__(self):
        return len(self.values)

    def __next__(self):
        curr_val = self.values[self._idx]
        if self._choosing_strategy == "cycle":
            self._idx = (self._idx + 1) % len(self)
        elif self._choosing_strategy == "random":
            self._idx = self.np_random.choice(np.arange(len(self)), p=self._probs)
        else:
            raise NotImplementedError
        return curr_val


class UniformRange(Range):
    def __init__(self, np_random, values):
        assert len(values) == 2, (
            "Must specify exactly 2 np.ndarray values ordered as [low, high]")
        assert values[0].shape == values[1].shape
        super(UniformRange, self).__init__(np_random, values)

    def __next__(self):
        return self.np_random.uniform(low=self.values[0], high=self.values[1])


class MultiUniformRange(Range):
    def __init__(self, np_random, values):
        for r in values:
            assert len(r) == 2
            assert r[0].shape == r[1].shape
        self.num_intervals = len(values)
        super(MultiUniformRange, self).__init__(np_random, values)

    def __next__(self):
        rand_idx = self.np_random.randint(self.num_intervals)
        rand_range = self.values[rand_idx]
        return self.np_random.uniform(low=rand_range[0], high=rand_range[1])

RANGES = {
    "DiscreteRange": DiscreteRange,
    "UniformRange": UniformRange,
    "MultiUniformRange": MultiUniformRange,
}

def get_range_from_params(params, np_random):
    range_type = params.pop("type", "DiscreteRange")
    return RANGES[range_type](np_random, **params)
