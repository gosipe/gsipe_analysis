# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:03:02 2023

@author: gsipe
"""

import numpy as np
from scipy.stats import sem
import sterr

def get_event_tc(master, idx_roi, event_loc, event_win, win_buffer, freq_samp):
    """
    Function that compares changes in tuned responses around events

    Args:
    - master: master data structure
    - idx_roi: indices of ROIs to analyze
    - event_loc: frame location of events (number of events x location)
    - event_win: time (s) before and after the event to analyze
    - win_buffer: buffer time (s) to exclude around the event
    - freq_samp: frequency of collected frames (Hz)

    Returns:
    - event_tc: dictionary containing event-related tuning curve information
      - idx: matrix indicating the trial and grating conditions for each event (trial number x grat number x condition)
      - tc: tuning curve responses for each ROI, grating, and condition (ROI x grat number x condition)
      - stats: standard error of the mean tuning curve responses for each condition (condition)

    """

    neuro_grat = master.analysis.neuro.grat
    grat_on = master.sess_info.grat.grat_on
    grat_off = master.sess_info.grat.grat_off
    grat_num = master.sess_info.grat.grat_num
    trial_num = master.sess_info.grat.trial_num

    buffer = win_buffer * freq_samp
    ori_matrix = neuro_grat.ori_matrix  # Matrix of frames when gratings are presented, rows = trials, columns=orientations
    num_roi = len(idx_roi)

    # Pre-allocate idx_ori for determining pupil/grating overlap
    idx_ori = {
        'pre': np.zeros((trial_num, grat_num), dtype=bool),
        'post': np.zeros((trial_num, grat_num), dtype=bool),
        'next': np.zeros((trial_num, grat_num), dtype=bool),
        'base': np.zeros((trial_num, grat_num), dtype=bool)
    }

    # Cycle through events and find which grating presentations they overlap with
    for e in range(event_loc.shape[0]):
        foo_pre = np.arange(event_loc[e] - buffer - event_win * freq_samp, event_loc[e] - buffer)
        foo_post = np.arange(event_loc[e] + buffer, event_loc[e] + buffer + event_win * freq_samp)
        # Cycle through grating number and trials
        for g in range(grat_num):
            for t in range(trial_num):
                foo_stim = np.arange(ori_matrix[t, g], ori_matrix[t, g] + grat_on * freq_samp)
                pre_overlap = np.intersect1d(foo_pre, foo_stim)
                post_overlap = np.intersect1d(foo_post, foo_stim)
                # Cutoff for frame overlap is >=0.5s (i.e., freq_samp/2)
                if len(pre_overlap) >= freq_samp / 2:
                    idx_ori['pre'][t, g] = True
                if len(post_overlap) >= freq_samp / 2:
                    idx_ori['post'][t, g] = True

    serial_ori = idx_ori['post'].T.flatten()
    serial_ori[-1] = 0
    serial_shift = np.roll(serial_ori, 1)
    idx_ori['next'] = serial_shift.reshape((grat_num, trial_num)).T
    idx_ori['base'] = ~(idx_ori['pre'] + idx_ori['post'])

    idx_matrix = np.zeros((trial_num, grat_num, 4), dtype=bool)
    idx_matrix[:, :, 0] = idx_ori['base']
    idx_matrix[:, :, 1] = idx_ori['pre']
    idx_matrix[:, :, 2] = idx_ori['post']
    idx_matrix[:, :, 3] = idx_ori['next']
    
    tc_matrix = np.zeros((num_roi, grat_num, 4))
    for r in range(num_roi):
        roi_tc = neuro_grat.roi[idx_roi[r]].resp.mean_r.diff
        base_tc = roi_tc.copy()
        base_tc[~idx_ori['base']] = np.nan
        base_mean = np.nanmean(base_tc, axis=0)
        delta_base_tc = roi_tc - base_mean
        foo_shift = neuro_grat.roi[idx_roi[r]].tc.mean_r.shift_val
        if foo_shift < 0:
            shift_idx = np.roll(idx_matrix, abs(foo_shift), axis=1)
            shift_tc = np.roll(delta_base_tc, abs(foo_shift), axis=1)
        elif foo_shift > 0:
            shift_idx = np.roll(idx_matrix, (grat_num - foo_shift), axis=1)
            shift_tc = np.roll(delta_base_tc, (grat_num - foo_shift), axis=1)
        else:
            shift_idx = idx_matrix.copy()
            shift_tc = delta_base_tc.copy()
        for j in range(4):
            for g in range(grat_num):
                tc_matrix[r, g, j] = np.nanmean(shift_tc[shift_idx[:, g, j], g], axis=0)
    
    tc_matrix = np.concatenate((tc_matrix[:, -1:, :], tc_matrix), axis=1)
    
    base_stats = sterr(np.squeeze(tc_matrix[:, :, 0]), axis=1)
    pre_stats = sterr(np.squeeze(tc_matrix[:, :, 1]), axis=1)
    post_stats = sterr(np.squeeze(tc_matrix[:, :, 2]), axis=1)
    next_stats = sterr(np.squeeze(tc_matrix[:, :, 3]), axis=1)
    
    event_tc = {}
    event_tc['idx'] = idx_matrix
    event_tc['tc'] = tc_matrix
    event_tc['stats'] = {
        'base': base_stats,
        'pre': pre_stats,
        'post': post_stats,
        'next': next_stats
    }

    return event_tc
                          
