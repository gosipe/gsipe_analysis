# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:40:17 2023

@author: Graybird
"""

import numpy as np
from scipy.signal import find_peaks

def find_event(event_data, freq_data, min_amp, deriv_thresh):
    # Setup and maintenance
    event_windows = []

    # Calculate derivative of high envelope and add zero
    event_dt = np.diff(event_data)
    event_dt = np.concatenate((event_dt, [0]))

    # Find peaks of envH to specify the overall events
    event_amp, event_loc = find_peaks(event_data, height=min_amp)

    # Find the beginning and end of the event using threshold and derivative
    for i in range(len(event_amp)):
        b1 = np.where((event_dt[:event_loc[i]] <= deriv_thresh) & (event_data[:event_loc[i]] <= 0))[0]
        b2 = event_loc[i] + np.where((event_dt[event_loc[i]:] >= deriv_thresh) & (event_data[event_loc[i]:] <= 0))[0][0]
        if b2 >= len(event_data):
            b2 = len(event_data) - 1
        if len(b1) > 0 and len(b2) > 0:
            event_windows.append([b1[-1], b2])

    # Find unique windows
    if len(event_windows) > 0:
        event_windows = np.array(event_windows)
        _, unique_indices = np.unique(event_windows[:, 1], return_index=True)
        wins = event_windows[unique_indices]
    else:
        wins = np.array([])

    if wins.size > 0:
        event_num = wins.shape[0]
        event_win = wins
        event_dur = (wins[:, 1] - wins[:, 0]) / freq_data
    else:
        event_num = 0
        event_win = np.array([])
        event_dur = np.array([])

    return event_num, event_win, event_dur
