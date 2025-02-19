# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:39:46 2023

@author: Graybird
"""

import numpy as np
from scipy.signal import find_peaks

def find_astro(astro_data, astro_fract, freq_astro, min_prom, diff_thresh, fract_thresh):
    astro_windows = []
    diff_vec = np.concatenate((np.diff(astro_data), [0]))
    astro_amp, astro_loc, _, astro_prom = find_peaks(astro_data, min_prominence=min_prom)

    for i in range(len(astro_amp)):
        b1 = np.where((diff_vec[:astro_loc[i]] <= diff_thresh) & (astro_data[:astro_loc[i]] <= astro_amp[i] - astro_prom[i] * 0.5))[0]
        b1 = b1[-1] if len(b1) > 0 else []
        b2 = astro_loc[i] + np.where((diff_vec[astro_loc[i]:] >= -diff_thresh) & (astro_data[astro_loc[i]:] <= astro_amp[i] - astro_prom[i] * 0.5))[0]
        b2 = b2[0] if len(b2) > 0 else len(astro_data)
        if not np.isnan(b1) and not np.isnan(b2):
            astro_windows.append([b1, b2])

    if len(astro_windows) > 0:
        _, _, aIndx = np.unique(astro_windows[:, 1], return_index=True, return_inverse=True)
        notdups = np.ones(len(astro_windows), dtype=bool)
        for j in range(len(aIndx) - 1):
            if aIndx[j] == aIndx[j + 1]:
                notdups[j + 1] = False
        wins = astro_windows[notdups]
    else:
        wins = np.nan

    if not np.isnan(wins):
        astro_fract_ct = []
        for i in range(wins.shape[0]):
            astro_act = np.max(astro_fract[:, wins[i, 0]:wins[i, 1]], axis=1)
            astro_prop_act = np.sum(astro_act, axis=0) / astro_act.shape[0]
            astro_fract_ct.append(astro_prop_act)
        astro_fract_ct = np.array(astro_fract_ct)
        pass_fract = np.where(astro_fract_ct >= fract_thresh)[0]
        lace_wins = wins[pass_fract]
    else:
        lace_wins = np.nan

    # Find the event statistics
    astro_amp = []
    astro_loc = []
    astro_prom = []
    if not np.isnan(lace_wins):
        for winEvent in range(lace_wins.shape[0]):
            amp, loc, _, prom = find_peaks(astro_data[lace_wins[winEvent, 0]:lace_wins[winEvent, 1]])
            astroMax = np.argmax(amp)
            astro_amp.append(amp[astroMax])
            astro_loc.append(loc[astroMax] + lace_wins[winEvent, 0] - 1)
            astro_prom.append(prom[astroMax])
        astro_dur = (lace_wins[:, 1] - lace_wins[:, 0]) / freq_astro
        astro_num = len(astro_amp)
        astro_freq = astro_num / (len(astro_data) / freq_astro)
        astro_win = lace_wins
    elif np.isnan(lace_wins).any() or lace_wins.size == 0:
        astro_amp = np.nan
        astro_loc = np.nan
        astro_win = np.nan
        astro_freq = np.nan
        astro_dur = np.nan
        astro_prom = np.nan
        astro_num = 0

    return astro_num, astro_freq, astro_amp, astro_prom, astro_loc, astro_dur, astro_win
