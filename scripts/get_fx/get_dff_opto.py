# -*- coding: utf-8 -*-
"""
Created on Tue May 16 19:52:20 2023

@author: gsipe
"""

import numpy as np
from scipy.stats import kde, zscore, percentileofscore

def get_dff_opto(f, real_frames, type='Method2'):
    # Determines if multiple ROIs are being analyzed
    if f.shape[0] > 1:
        num_roi = f.shape[0]
    else:
        num_roi = 1

    # Preallocate matrices
    dff = np.zeros((num_roi, f.shape[1]))
    f_base = np.zeros(num_roi)

    # Switch method and perform DF/F computation
    if type == 'Method1':
        f_prct = 5  # set percentile for calculating the base threshold
        for roi in range(num_roi):
            f_base[roi] = np.percentile(f[roi, real_frames], f_prct)
            if f_base[roi] <= 1:
                f[roi, :] = f[roi, :] + np.abs(f_base[roi])
                f_base[roi] = np.abs(f_base[roi])
            dff[roi, :] = (f[roi, :] - f_base[roi]) / f_base[roi]
        final_dff = dff
    elif type == 'Method2':
        for roi in range(num_roi):
            ksd = kde.gaussian_kde(f[roi, real_frames])
            xi = np.linspace(np.min(f[roi, real_frames]), np.max(f[roi, real_frames]), 100)
            ksd_values = ksd(xi)
            max_idx = np.argmax(ksd_values)
            f_0 = xi[max_idx]
            dff[roi, :] = (f[roi, :] - f_0) / f_0
        final_dff = dff
    elif type == 'Method3':
        for roi in range(num_roi):
            dff[roi, :] = zscore(f[roi, :], axis=1)
        final_dff = dff

    return final_dff
