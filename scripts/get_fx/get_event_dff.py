# -*- coding: utf-8 -*-
"""
Created on Tue May 16 19:54:02 2023

@author: gsipe
"""

import numpy as np
from scipy.stats import sem

def get_event_dff(dff, event_loc, event_win, freq_samp):
    """
    Function that calculates activity around events (rows x columns)

    Args:
    - dff: matrix of activity (ROIs x frames)
    - event_loc: frame location of event (number events x location)
    - event_win: time (s) before and after event to analyze
    - freq_samp: frequency of collected frames (Hz)

    Returns:
    - event_dff: dictionary containing event-related information
      - act: activity for each event (ROIs x event frames x event number)
      - event_avg: average activity across events (ROIs x event frames)
      - roi_avg: average activity across ROIs (event frames)
      - sterr: standard error of the mean across ROIs (event frames)
      - diff: difference in activity between before and after each event (ROIs x event number)
      - sort_idx: sorted indices of ROIs based on the difference in activity

    """

    if len(event_win) == 1:
        event_win = [event_win, event_win]

    num_roi = dff.shape[0]  # Find number of ROIs
    num_frame = dff.shape[1]  # Find number of frames

    frame_side = np.round(event_win * freq_samp).astype(int)  # Calculate number of frames on each side of the event
    event_frames = np.sum(frame_side) + 1  # Calculate number of frames before and after the event

    # Remove events where the windows are outside the frame bounds
    valid_loc = event_loc[
        ~((event_loc - frame_side[0] <= 0) | (event_loc + frame_side[1] > num_frame))
    ]

    event_num = len(valid_loc)  # Calculate remaining number of events
    event_act = np.zeros((num_roi, event_frames, event_num))  # Pre-allocate activity variable

    # Extract the relevant activity from each ROI for each event
    for i, loc in enumerate(valid_loc):
        event_act[:, :, i] = dff[:, loc - frame_side[0]:loc + frame_side[1] + 1]

    event_dff = {}  # Create a dictionary to store the results

    event_dff['act'] = event_act  # Save all data
    event_dff['event_avg'] = np.mean(event_act, axis=2)
    event_dff['roi_avg'] = np.mean(event_act, axis=0)
    event_dff['sterr'] = sem(np.squeeze(np.mean(event_act, axis=0)), axis=0)

    # Find difference between before and after
    event_diff = np.zeros((num_roi, event_num))

    for i in range(event_num):
        event_diff[:, i] = np.mean(event_act[:, :frame_side[0], i], axis=1) - np.mean(event_act[:, frame_side[1]:, i], axis=1)

    # Sort by highest difference
    mean_diff = np.mean(event_diff, axis=1)
    sort_idx = np.argsort(mean_diff)[::-1]
    roi_mean = np.mean(event_act, axis=2)[sort_idx]

    event_dff['diff'] = event_diff
    event_dff['sort_idx'] = sort_idx

    return event_dff
