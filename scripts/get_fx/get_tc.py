# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:26:03 2023

@author: gsipe
"""

import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import circshift
import sterr
import VMFit
import VonMisesFunction

def get_tc(vis_data, grat_on, grat_off, neuro_freq, pval_cutoff):
    """
    Computes the tuning curves and responses for a given visual stimulus.

    Args:
    - vis_data: 3D array of visual data (num_frames x num_trial x num_grat)
    - grat_on: duration of grating on period in seconds
    - grat_off: duration of grating off period in seconds
    - neuro_freq: neuronal frequency in Hz
    - pval_cutoff: p-value threshold for t-test

    Returns:
    - tc: dictionary containing tuning curve information
    - resp: dictionary containing response information

    """

    num_grat = vis_data.shape[2]
    num_trial = vis_data.shape[1]

    off_last_sec = grat_off - 1
    on_last_sec = grat_on - 1
    off_frames = np.arange(off_last_sec * neuro_freq, neuro_freq * grat_off)
    on_frames = np.arange(
        neuro_freq * grat_off + on_last_sec * neuro_freq,
        neuro_freq * grat_off + neuro_freq * grat_on
    )

    delta_deg = 360 / num_grat
    ori_vec = np.arange(0, 360 - 360 / delta_deg, delta_deg)

    resp = {}
    resp['mean_r'] = {}
    resp['mean_r']['on'] = np.zeros((num_trial, num_grat))
    resp['mean_r']['off'] = np.zeros((num_trial, num_grat))
    resp['mean_r']['diff'] = np.zeros((num_trial, num_grat))
    resp['max_r'] = {}
    resp['max_r']['on'] = np.zeros((num_trial, num_grat))
    resp['max_r']['off'] = np.zeros((num_trial, num_grat))
    resp['max_r']['diff'] = np.zeros((num_trial, num_grat))
    resp['mean_r']['test'] = {}
    resp['mean_r']['test']['idx'] = np.zeros((1, num_grat))
    resp['mean_r']['test']['pval'] = np.zeros((1, num_grat))
    resp['max_r']['test'] = {}
    resp['max_r']['test']['idx'] = np.zeros((1, num_grat))
    resp['max_r']['test']['pval'] = np.zeros((1, num_grat))

    for ori in range(num_grat):
        resp['max_r']['off'][:, ori] = np.squeeze(np.max(vis_data[off_frames, :, ori], axis=0))
        resp['max_r']['on'][:, ori] = np.squeeze(np.max(vis_data[on_frames, :, ori], axis=0))
        resp['max_r']['diff'][:, ori] = resp['max_r']['on'][:, ori] - resp['max_r']['off'][:, ori]
        _, pval = ttest_ind(resp['max_r']['on'][:, ori], resp['max_r']['off'][:, ori])
        resp['max_r']['test']['idx'][0, ori] = pval < pval_cutoff
        resp['max_r']['test']['pval'][0, ori] = pval

        resp['mean_r']['off'][:, ori] = np.squeeze(np.mean(vis_data[off_frames, :, ori], axis=0))
        resp['mean_r']['on'][:, ori] = np.squeeze(np.mean(vis_data[on_frames, :, ori], axis=0))
        resp['mean_r']['diff'][:, ori] = resp['mean_r']['on'][:, ori] - resp['mean_r']['off'][:, ori]
        Args_, pval = ttest_ind(resp['mean_r']['on'][:, ori], resp['mean_r']['off'][:, ori])
        resp['mean_r']['test']['idx'][0, ori] = pval < pval_cutoff
        resp['mean_r']['test']['pval'][0, ori] = pval
        resp['mean_r']['norm'] = (resp['mean_r']['diff'] - np.min(resp['mean_r']['diff'], axis=1, keepdims=True)) / (
        np.max(resp['mean_r']['diff'], axis=1, keepdims=True) - np.min(resp['mean_r']['diff'], axis=1, keepdims=True)
        )
        resp['max_r']['norm'] = (resp['max_r']['diff'] - np.min(resp['max_r']['diff'], axis=1, keepdims=True)) / (
                np.max(resp['max_r']['diff'], axis=1, keepdims=True) - np.min(resp['max_r']['diff'], axis=1, keepdims=True)
        )
        
        resp['max_r']['diff_stat'] = sterr(resp['max_r']['diff'], axis=1)
        resp['mean_r']['diff_stat'] = sterr(resp['mean_r']['diff'], axis=1)
        resp['max_r']['norm_stat'] = sterr(resp['max_r']['norm'], axis=1)
        resp['mean_r']['norm_stat'] = sterr(resp['mean_r']['norm'], axis=1)
        
        tc = {}
        tc['max_r'] = {}
        tc['mean_r'] = {}
        tc['max_r']['diff'] = resp['max_r']['diff_stat']['mean']
        tc['mean_r']['diff'] = resp['mean_r']['diff_stat']['mean']
        tc['max_r']['norm'] = resp['max_r']['norm_stat']['mean']
        tc['mean_r']['norm'] = resp['mean_r']['norm_stat']['mean']
        
        loc_max_max = np.argmax(tc['max_r']['diff'])
        tc['max_r']['pref_grat'] = ori_vec[loc_max_max]
        loc_mean_max = np.argmax(tc['mean_r']['diff'])
        tc['mean_r']['pref_grat'] = ori_vec[loc_mean_max]
        
        tc['mean_r']['diff_norm'] = normalize(tc['mean_r']['diff'], axis=1)
        tc['max_r']['diff_norm'] = normalize(tc['max_r']['diff'], axis=1)
        
        mid_ori = num_grat // 2
        
        tc['max_r']['shift_val'] = loc_max_max - mid_ori
        tc['mean_r']['shift_val'] = loc_mean_max - mid_ori
        
        if tc['max_r']['shift_val'] < 0:
            tc['max_r']['diff_shift'] = circshift(tc['max_r']['diff'], abs(tc['max_r']['shift_val']), axis=1)
            tc['max_r']['norm_shift'] = circshift(tc['max_r']['norm'], abs(tc['max_r']['shift_val']), axis=1)
            resp['max_r']['diff_shift'] = circshift(resp['max_r']['diff'], abs(tc['max_r']['shift_val']), axis=1)
            resp['max_r']['norm_shift'] = circshift(resp['max_r']['norm'], abs(tc['max_r']['shift_val']), axis=1)
        elif tc['max_r']['shift_val'] > 0:
            tc['max_r']['diff_shift'] = circshift(tc['max_r']['diff'], num_grat - tc['max_r']['shift_val'], axis=1)
            tc['max_r']['norm_shift'] = circshift(tc['max_r']['norm'], num_grat - tc['max_r']['shift_val'], axis=1)
            resp['max_r']['diff_shift'] = circshift(resp['max_r']['diff'], num_grat - tc['max_r']['shift_val'], axis=1)
            resp['max_r']['norm_shift'] = circshift(resp['max_r']['norm'], num_grat - tc['max_r']['shift_val'], axis=1)
        else:
            tc['max_r']['diff_shift'] = tc['max_r']['diff']
            tc['max_r']['norm_shift'] = tc['max_r']['norm']
            resp['max_r']['diff_shift'] = resp['max_r']['diff']
            resp['max_r']['norm_shift'] = resp['max_r']['norm']
        if tc['mean_r']['shift_val'] < 0:
            tc['mean_r']['diff_shift'] = circshift(tc['mean_r']['diff'], abs(tc['mean_r']['shift_val']), axis=1)
            tc['mean_r']['norm_shift'] = circshift(tc['mean_r']['norm'], abs(tc['mean_r']['shift_val']), axis=1)
            resp['mean_r']['diff_shift'] = circshift(resp['mean_r']['diff'], abs(tc['mean_r']['shift_val']), axis=1)
            resp['mean_r']['norm_shift'] = circshift(resp['mean_r']['norm'], abs(tc['mean_r']['shift_val']), axis=1)
        elif tc['mean_r']['shift_val'] > 0:
            tc['mean_r']['diff_shift'] = circshift(tc['mean_r']['diff'], num_grat - tc['mean_r']['shift_val'], axis=1)
            tc['mean_r']['norm_shift'] = circshift(tc['mean_r']['norm'], num_grat - tc['mean_r']['shift_val'], axis=1)
            resp['mean_r']['diff_shift'] = circshift(resp['mean_r']['diff'], num_grat - tc['mean_r']['shift_val'], axis=1)
            resp['mean_r']['norm_shift'] = circshift(resp['mean_r']['norm'], num_grat - tc['mean_r']['shift_val'], axis=1)
        else:
            tc['mean_r']['diff_shift'] = tc['mean_r']['diff']
            tc['mean_r']['norm_shift'] = tc['mean_r']['norm']
            resp['mean_r']['diff_shift'] = resp['mean_r']['diff']
            resp['mean_r']['norm_shift'] = resp['mean_r']['norm']
        
        tc['info']['angle_vec'] = ori_vec
        
        angles_deg = np.linspace(0, 360 - delta_deg, 20000)
        angles_rads = angles_deg * (np.pi / 180)
        
        coeff_set, good_fit = VMFit(tc['max_r']['diff_norm'], tc['max_r']['pref_grat'])
        tc['max_r']['vm_fit'] = good_fit
        tc['max_r']['vm_coeff'] = coeff_set
        tc['max_r']['vm_fx'] = VonMisesFunction(coeff_set, angles_rads)
        
        coeff_set, good_fit = VMFit(tc['mean_r']['diff_norm'], tc['mean_r']['pref_grat'])
        tc['mean_r']['vm_fit'] = good_fit
        tc['mean_r']['vm_fx'] = coeff_set
        tc['mean_r']['vm_fx'] = VonMisesFunction(coeff_set, angles_rads)
        
        return tc, resp

