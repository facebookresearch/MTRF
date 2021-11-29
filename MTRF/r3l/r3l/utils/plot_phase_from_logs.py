# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import numpy as np
from vtils.plotting.simple_plot import *
from scipy import signal

def get_phase(filename):
    try:
        data = np.genfromtxt(filename, dtype=str, delimiter=',', skip_header=1, usecols=(14))
    except Exception as e:
        print("WARNING: %s not found." % filename)
    return data

def smooth_data(y, window_length=101, polyorder=3):
    window_length = min(int(len(y) / 2),
                        window_length)  # set maximum valid window length
    # if window not off
    if window_length % 2 == 0:
        window_length = window_length + 1
    return signal.savgol_filter(y, window_length, polyorder)

file_name = '/home/vik/Projects/r3l/manipulate/SawyerDhandInHandManipulateResetFree-v0-EJ0e_0/logs/log.csv'
phase_graph = get_phase(file_name)

phase0 = []
phase1 = []
phase2 = []
phase3 = []
phase4 = []

for phase_iter in phase_graph:
    phase0.append(phase_iter.count('0')*33)
    phase1.append(phase_iter.count('1')*33)
    phase2.append(phase_iter.count('2')*33)
    phase3.append(phase_iter.count('3')*33)
    phase4.append(phase_iter.count('4')*33)

plot(smooth_data(phase0), legend='Rotate', fig_name="PhaseGraph")
plot(smooth_data(phase1), legend='Flip_D', fig_name="PhaseGraph")
plot(smooth_data(phase2), legend='Reach', fig_name="PhaseGraph")
plot(smooth_data(phase3), legend='PickUp', fig_name="PhaseGraph")
plot(smooth_data(phase4), legend='Flip_U', fig_name="PhaseGraph",
    yaxislabel="% time spent in phase during eval",
    xaxislabel="epochs",
    plot_name="Phase transitions during eval")
show_plot()
