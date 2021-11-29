# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import numpy as np

def circle_distance(euler1, euler2):
    """Always <= np.pi radians"""
    euler1 = np.mod(euler1, 2 * np.pi)
    euler2 = np.mod(euler2, 2 * np.pi)
    abs_diff = np.abs(euler1 - euler2)
    return np.minimum(abs_diff, 2 * np.pi - abs_diff)

def circle_distance_mod(euler1, euler2, mod=2*np.pi/3):
    euler1 = np.mod(euler1, mod)
    euler2 = np.mod(euler2, mod)
    abs_diff = np.abs(euler1 - euler2)
    return np.minimum(abs_diff, mod - abs_diff)

def quat_distance(quat1, quat2):
    """ Maps to [0, 1].
        Implements Phi_4 from http://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf"""
    return 1 - np.abs(np.dot(quat1, quat2))
