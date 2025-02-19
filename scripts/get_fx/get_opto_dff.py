# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:11:18 2023

@author: gsipe
"""

import numpy as np
from scipy.stats import sem

def get_opto_dff(master, act_win):
    """
    Calculates activity around optogenetic stimulation.

    Args:
    - master: data master object (or dictionary) containing the required fields
    - act_win: window size in seconds after stimulation to analyze (float)

    Returns:
    - opto_dff: dictionary containing the optogenetic stimulation data

    """

    idx_roi = master.analysis.neuro.grat.osi >= 0.4
    dff = master.data.neuro.dff[idx_roi, :]
    stim_end = master.sess_info.opto.stim_end
    base_end = master.sess_info.opto.base_end
    stim1_end = stim_end[:, 0]
    stim2_end = stim_end[:, 1]

    freq_neuro = master.data.neuro.info.freq_neuro
    num_roi = dff.shape[0]  # Find number of ROIs
    num_frame = dff.shape[1]  # Find number of frames
    frame_win = int(act_win * freq_neuro)  # Calculate number of frames after stimulation to analyze

    # Remove events where the windows are outside the frame bounds
    valid_stim1 = stim1_end[~(stim1_end + frame_win > num_frame)]
    valid_stim2 = stim2_end[~(stim2_end + frame_win > num_frame)]
    valid_base = base_end[~(base_end + frame_win > num_frame)]

    # Process responses to base trials
    base_event_num = valid_base.shape[0]  # Calculate remaining number of events
    base_dff = np.zeros((num_roi, frame_win, base_event_num))  # Pre-allocate activity variable (ROIs x event frames x event number)
    # Extract the relevant activity from each ROI for each event
    for i in range(base_event_num):
        base_dff[:, :, i] = dff[:, valid_base[i] + 1 : valid_base[i] + frame_win]
    opto_dff = {"base": {"act": base_dff, "avg": np.mean(base_dff, axis=2), "sterr": sem(np.mean(base_dff, axis=2), axis=1)}}

    # Process responses to first stim type
    stim1_event_num = valid_stim1.shape[0]  # Calculate remaining number of events
    stim1_dff = np.zeros((num_roi, frame_win, stim1_event_num))  # Pre-allocate activity variable (ROIs x event frames x event number)
    # Extract the relevant activity from each ROI for each event
    for i in range(stim1_event_num):
        stim1_dff[:, :, i] = dff[:, valid_stim1[i] + 1 : valid_stim1[i] + frame_win]
    opto_dff["stim"] = [{"act": stim1_dff, "avg": np.mean(stim1_dff, axis=2), "sterr": sem(np.mean(stim1_dff, axis=2), axis=1)}]

    # Process responses to second stim type
    stim2_event_num = valid_stim2.shape[0]  # Calculate remaining number of events
    stim2_dff = np.zeros((num_roi, frame_win, stim2_event_num))  # Pre-allocate activity variable (ROIs x event frames x event number)
    # Extract the relevant activity from each ROI for each event
    for i in range(stim2_event_num):
        stim2_dff[:, :, i] = dff[:, valid_stim2[i] + 1 : valid_stim2[i] + frame_win]
        opto_dff["stim"].append({"act": stim2_dff, "avg": np.mean(stim2_dff, axis=2), "sterr": sem(np.mean(stim2_dff, axis=2), axis=1)})
    
    return opto_dff
    