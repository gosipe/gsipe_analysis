# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:07:39 2023

@author: gsipe
"""

import numpy as np

def get_gain(trial_data):
    """
    Function that takes in calcium data sorted as trials x time (rows x columns),
    including the event locations as trial/time pairs, and the size of the window to analyze.

    Args:
    - trial_data: calcium data sorted as trials x time (2D array)

    Returns:
    - trial_gain: calcium data scaled and centered based on the mean of each trial (2D array)
    - trace_gain: flattened and reshaped version of trial_gain (1D array)

    """

    trial_scale = trial_data * 100
    trial_min = np.min(np.min(trial_scale))

    if 0 <= trial_min < 1:
        dff_offset = trial_min + 1
        trial_offset = trial_scale + dff_offset
    elif trial_min < 0:
        dff_offset = np.abs(trial_min) + 1
        trial_offset = trial_scale + dff_offset
    else:
        trial_offset = trial_scale
        mean_offset = trial_min

    trial_num, trial_time = trial_offset.shape
    trial_gain = np.zeros((trial_num, trial_time))

    trial_mean = np.mean(trial_offset, axis=0)
    for trial in range(trial_num):
        trial_gain[trial, :] = (trial_offset[trial, :] - trial_mean) / trial_mean

    trace_gain = trial_gain.T.flatten()

    return trial_gain, trace_gain
