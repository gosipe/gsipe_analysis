# -*- coding: utf-8 -*-
"""
Created on Tue May 16 19:56:06 2023

@author: gsipe
"""

import numpy as np
from scipy.stats import sem
import get_gain

def get_event_gain(neuro_trial, event_loc, event_win, freq_samp):
    """
    Function that takes in calcium data sorted as trials x time (rows x columns)
    including the event locations as trial/time pairs, and the size of window to analyze

    Args:
    - neuro_trial: calcium data sorted as trials x time (rows x columns)
    - event_loc: event locations as trial/time pairs
    - event_win: size of the window to analyze
    - freq_samp: frequency of collected frames (Hz)

    Returns:
    - event_gain: dictionary containing event-related information
      - all_gain: gain traces for each event (ROIs x window frames x event number)
      - all_dff: dF/F traces for each event (ROIs x window frames x event number)
      - gain: dictionary containing gain information
        - neuro_mean: mean gain trace across all events (window frames)
        - event_mean: mean gain trace for each event (event frames)
        - stats: standard error of the mean gain trace (window frames)
      - dff: dictionary containing dF/F information
        - neuro_mean: mean dF/F trace across all events (window frames)
        - event_mean: mean dF/F trace for each event (event frames)

    """

    max_frames = neuro_trial.shape[1] * neuro_trial.shape[2]
    num_roi = neuro_trial.shape[0]
    trial_frame = neuro_trial.shape[1]
    num_trials = neuro_trial.shape[2]

    event_num = event_loc.shape[0]
    valid_event = np.ones(event_num, dtype=bool)

    win_frame = 1 + event_win * 2 * freq_samp
    foo_all = np.zeros((num_roi, win_frame))

    for e in range(event_num):
        if event_loc[e] - event_win * freq_samp <= 0 or event_loc[e] + event_win * freq_samp > max_frames:
            valid_event[e] = False

    event_loc = event_loc[valid_event]
    event_num = event_loc.shape[0]

    all_gain = np.zeros((num_roi, win_frame, event_num))
    all_dff = np.zeros((num_roi, win_frame, event_num))

    for r in range(num_roi):
        foo_roi = np.transpose(np.squeeze(neuro_trial[r, :, :]))
        foo_reshape = np.reshape(foo_roi.T, (-1,), order='F')
        _, trace_gain = get_gain(foo_roi)

        for e in range(event_num):
            all_gain[r, :, e] = trace_gain[event_loc[e] - event_win * freq_samp:event_loc[e] + event_win * freq_samp + 1]
            all_dff[r, :, e] = foo_reshape[event_loc[e] - event_win * freq_samp:event_loc[e] + event_win * freq_samp + 1]

    event_gain = {}  # Create a dictionary to store the results

    event_gain['all_gain'] = all_gain
    event_gain['all_dff'] = all_dff

    event_gain['gain'] = {}
    event_gain['dff'] = {}

    event_gain['gain']['neuro_mean'] = np.transpose(np.squeeze(np.mean(all_gain, axis=2)))
    event_gain['dff']['neuro_mean'] = np.transpose(np.squeeze(np.mean(all_dff, axis=2)))
    event_gain['gain']['event_mean'] = np.squeeze(np.mean(all_gain, axis=2))
    event_gain['dff']['event_mean'] = np.squeeze(np.mean(all_dff, axis=2))
    event_gain['gain']['stats'] = sem(event_gain['gain']['neuro_mean'], axis=0)

    return event_gain
    
