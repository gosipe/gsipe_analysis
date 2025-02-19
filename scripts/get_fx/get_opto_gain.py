# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:15:07 2023

@author: gsipe
"""

import numpy as np

def get_opto_gain(foo_roi):
    """
    Calculates gain in calcium data for optogenetic stimulation.

    Args:
    - foo_roi: dictionary containing the calcium data for base and stimulation trials

    Returns:
    - stim1_gain: gain values for stimulation 1 trials (numpy array)
    - stim2_gain: gain values for stimulation 2 trials (numpy array)

    """

    stim_num = foo_roi["stim"][0]["trial"].shape[0]

    foo_base = foo_roi["base"]["trial"] * 100
    foo_stim1 = foo_roi["stim"][0]["trial"] * 100
    foo_stim2 = foo_roi["stim"][1]["trial"] * 100

    base_min = np.min(np.min(foo_base))
    if base_min < 1 and base_min >= 0:
        dff_offset = base_min + 1
        base_offset = foo_base + dff_offset
    elif base_min < 0:
        dff_offset = abs(base_min) + 1
        base_offset = foo_base + dff_offset
    else:
        base_offset = foo_base

    stim1_min = np.min(np.min(foo_stim1))
    if stim1_min < 1 and stim1_min >= 0:
        dff_offset = stim1_min + 1
        stim1_offset = foo_stim1 + dff_offset
    elif stim1_min < 0:
        dff_offset = abs(stim1_min) + 1
        stim1_offset = foo_stim1 + dff_offset
    else:
        stim1_offset = foo_stim1

    stim2_min = np.min(np.min(foo_stim2))
    if stim2_min < 1 and stim2_min >= 0:
        dff_offset = stim2_min + 1
        stim2_offset = foo_stim2 + dff_offset
    elif stim2_min < 0:
        dff_offset = abs(stim2_min) + 1
        stim2_offset = foo_stim2 + dff_offset
    else:
        stim2_offset = foo_stim1

    stim1_gain = np.zeros((8, 200))
    stim2_gain = np.zeros((8, 200))
    base_mean = np.mean(base_offset, axis=0)
    for trial in range(stim_num):
        stim1_gain[trial, :] = (stim1_offset[trial, :] - base_mean) / base_mean
        stim2_gain[trial, :] = (stim2_offset[trial, :] - base_mean) / base_mean

    return stim1_gain, stim2_gain
