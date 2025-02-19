# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:00:43 2023

@author: gsipe
"""

import numpy as np
from scipy.stats import sem
import get_pw_corr

def get_event_pwcorr(dff, event_loc, event_win, corr_bin, freq_samp, partial_data=None):
    """
    Function that calculates pair-wise correlations between ROIs around events with the possibility of taking into account correlation with the event

    Args:
    - dff: matrix of activity (ROIs x frames)
    - event_loc: frame location of events (number of events x location)
    - event_win: time (s) before and after the event to analyze
    - corr_bin: number of frames to compare for each correlation timepoint
    - freq_samp: frequency of collected frames (Hz)
    - partial_data: partial data to correct with partial pair-wise correlations (optional)

    Returns:
    - event_pwcorr: dictionary containing event-related pairwise correlation information
      - data: pairwise correlation values for each event (event number x event frames)
      - stats: standard error of the mean pairwise correlation values (event frames)

    """

    if partial_data is None:
        partial_flag = False
    elif not np.isnan(partial_data):
        partial_flag = True
    else:
        partial_flag = False

    num_roi = dff.shape[0]  # Find number of ROIs
    num_frame = dff.shape[1]  # Find number of frames
    frame_side = event_win * freq_samp  # Calculate number of frames on each side of the event
    event_frames = frame_side * 2 + 1  # Calculate number of frames before and after the event

    # Remove events where the windows are outside the frame bounds
    valid_loc = event_loc[~(event_loc - event_win * freq_samp - corr_bin <= 0) & ~(event_loc + event_win * freq_samp + corr_bin >= num_frame)]
    event_num = valid_loc.shape[0]  # Calculate the remaining number of events

    # Pre-allocate activity variable (event number x event frames)
    event_pw = np.zeros((event_num, event_frames))
    slide = np.arange(event_frames)

    if partial_flag:
        itx = 0
        for e in range(event_num):
            start_idx = valid_loc[itx] - frame_side
            for i in range(event_frames):
                corrwin = np.arange(start_idx - corr_bin + slide[i], start_idx + corr_bin + slide[i] + 1)
                _, _, _, avg_corr = get_pw_corr(dff[:, corrwin], partial_data[0, corrwin])
                event_pw[itx, i] = avg_corr['mean']
            itx += 1
    else:
        itx = 0
        for e in range(event_num):
            start_idx = valid_loc[itx] - frame_side
            for i in range(event_frames):
                corrwin = np.arange(start_idx - corr_bin + slide[i], start_idx + corr_bin + slide[i] + 1)
                _, _, _, avg_corr = get_pw_corr(dff[:, corrwin])
                event_pw[itx, i] = avg_corr['mean']
            itx += 1

    event_pwcorr = {}  # Create a dictionary to store the results

    event_pwcorr['data'] = event_pw
    event_pwcorr['stats'] = sem(event_pwcorr['data'], axis=0)

    return event_pwcorr
