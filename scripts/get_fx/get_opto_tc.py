# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:16:37 2023

@author: gsipe
"""

import numpy as np
import sterr

def get_opto_tc(master):
    """
    Compares changes in tuned responses around optogenetic stimulations.

    Args:
    - master: dictionary containing grating responses and analysis parameters

    Returns:
    - opto_tc: dictionary containing the neuronal population mean, standard error,
               index matrix, and tuning curve matrix

    """

    neuro_grat = master["analysis"]["neuro"]["grat"]
    opto_params = master["sess_info"]["opto"]
    freq_neuro = master["data"]["neuro"]["info"]["freq_neuro"]
    grat_on = master["sess_info"]["grat"]["grat_on"]
    grat_off = master["sess_info"]["grat"]["grat_off"]
    grat_num = master["sess_info"]["grat"]["grat_num"]
    trial_num = master["sess_info"]["grat"]["trial_num"]

    ori_matrix = neuro_grat["ori_matrix"]

    stim_grat = opto_params["optostimtrial"]
    stim_freq = opto_params["optoTrialFreq"]
    stim_type = opto_params["stimType"]
    stim_type_num = len(stim_type)

    idx_ori = {
        "pre": np.zeros((trial_num, grat_num), dtype=bool),
        "post": np.zeros((trial_num, grat_num), dtype=bool)
    }
    idx_roi = np.arange(master["analysis"]["neuro"]["grat"]["info"]["num_roi"])
    num_roi = len(idx_roi)

    idx_matrix = np.zeros((trial_num, grat_num, 5))
    trial_num_ct = 1
    stim_num_ct = 1
    for t in range(trial_num):
        stim_trial = trial_num_ct % stim_freq
        if stim_trial:
            if stim_num_ct == 1:
                idx_matrix[t, stim_grat - 1, 1] = 1
                foo_next = stim_grat
                if foo_next > grat_num:
                    idx_matrix[t + 1, 0, 2] = 1
                else:
                    idx_matrix[t, stim_grat, 2] = 1
            elif stim_num_ct == 2:
                idx_matrix[t, stim_grat - 1, 3] = 1
                foo_next = stim_grat
                if foo_next > grat_num:
                    idx_matrix[t + 1, 0, 4] = 1
                else:
                    idx_matrix[t, stim_grat, 4] = 1
            stim_num_ct += 1
            if stim_num_ct > stim_type_num:
                stim_num_ct = 1
        trial_num_ct += 1
    idx_matrix[:, :, 0] = ~(idx_matrix[:, :, 1] + idx_matrix[:, :, 2] + idx_matrix[:, :, 3] + idx_matrix[:, :, 4])

    tc_matrix = np.zeros((num_roi, grat_num, 5))
    non_base = (idx_matrix[:, :, 1] + idx_matrix[:, :, 2] + idx_matrix[:, :, 3] + idx_matrix[:, :, 4])
    tc_matrix[:, :, 0] = np.nan

    for r in range(num_roi):
        foo_roi = neuro_grat["roi"][idx_roi[r]]["resp"]["mean_r"]["shift"]
        foo_mean = np.mean(foo_roi, axis=0)
        foo_roi = foo_roi - foo_mean
        foo_shift = neuro_grat["roi"][idx_roi[r]]["tc"]["mean_r"]["shift_val"]
        if foo_shift < 0:
            shift_idx= np.roll(idx_matrix, abs(foo_shift), axis=1)
        elif foo_shift > 0:
            shift_idx = np.roll(idx_matrix, (grat_num - foo_shift), axis=1)
        else:
            shift_idx = idx_matrix
        for j in range(5):
            for g in range(grat_num):
                tc_matrix[r, g, j] = np.nanmean(foo_roi[np.where(shift_idx[:, g, j]), g])
                tc_matrix = np.concatenate((tc_matrix[:, -1, :], tc_matrix), axis=1)
        opto_tc = {}
        opto_tc["base"] = {"stats": sterr(np.squeeze(tc_matrix[:, :, 0]), axis=1)}
        opto_tc["stim"] = [    {"stim": {"stats": sterr(np.squeeze(tc_matrix[:, :, 1]), axis=1)}},
            {"next": {"stats": sterr(np.squeeze(tc_matrix[:, :, 2]), axis=1)}}
        ]
        opto_tc["stim"].append(
            {"stim": {"stats": sterr(np.squeeze(tc_matrix[:, :, 3]), axis=1)}},
            {"next": {"stats": sterr(np.squeeze(tc_matrix[:, :, 4]), axis=1)}}
        )
        opto_tc["num_roi"] = num_roi
        opto_tc["idx"] = idx_matrix
        opto_tc["tc"] = tc_matrix
        
        return opto_tc
    
