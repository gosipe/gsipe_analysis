# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:46:20 2023

@author: Graybird
"""

import numpy as np
from scipy.signal import find_peaks

def find_pupil(pupil_data, freq_pupil, min_prom, diff_thresh):
    pupil_windows = []

    pupil_smooth = np.transpose(np.convolve(pupil_data, np.ones(freq_pupil*2), mode='same') / (freq_pupil*2))
    pupil_diff = np.concatenate((np.diff(pupil_smooth), [0]))

    # Find peaks of data vector to specify the overall events
    p_amps, p_locs, _, p_proms = find_peaks(pupil_smooth, prominence=min_prom)

    # Find the beginning and end of the event using threshold and derivative
    for i in range(len(p_amps)):
        b1 = np.where((pupil_diff[:p_locs[i]] <= diff_thresh) & (pupil_data[:p_locs[i]] <= p_amps[i] - p_proms[i]*0.5))[0]
        b2 = p_locs[i] + np.where((pupil_diff[p_locs[i]:] >= diff_thresh) & (pupil_data[p_locs[i]:] <= p_amps[i] - p_proms[i]*0.5))[0][0]
        if b2 >= len(pupil_data):
            b2 = len(pupil_data) - 1
        if len(b1) > 0 and len(b2) > 0:
            pupil_windows.append([b1[-1], b2])

    # Find unique windows
    if len(pupil_windows) > 0:
        pupil_windows = np.array(pupil_windows)
        _, unique_indices = np.unique(pupil_windows[:, 1], return_index=True)
        wins = pupil_windows[unique_indices]
    else:
        wins = np.array([])

    # Find the event statistics
    pupil_amp = []
    pupil_loc = []
    pupil_peak = []
    pupil_prom = []
    if np.isnan(wins):
        pupil_amp = np.array([])
        pupil_loc = np.array([])
        pupil_peak = np.array([])
        pupil_dur = np.array([])
        pupil_num = 0
        pupil_win = np.array([])
        pupil_prom = np.array([])
    elif not np.isnan(wins):
        val_win_index = np.ones(wins.shape[0], dtype=bool)
        for win_event in range(wins.shape[0]):
            amp, loc, _, prom = find_peaks(pupil_data[wins[win_event, 0]:wins[win_event, 1]])
            if len(amp) > 0:
                eveMax = np.argmax(amp)
                pupil_amp.append(amp[eveMax])
                pupil_loc.append(loc[eveMax] + wins[win_event, 0])
                pupil_peak.append(len(amp))
                pupil_prom.append(prom[eveMax])
            elif len(amp) == 0:
                val_win_index[win_event] = False
        wins = wins[val_win_index]

        pupil_dur = (wins[:, 1] - wins[:, 0]) / freq_pupil
        pupil_num = len(pupil_amp)
        pupil_win = wins

    return pupil_num, pupil_amp, pupil_prom, pupil_loc, pupil_peak, pupil_win, pupil_dur
