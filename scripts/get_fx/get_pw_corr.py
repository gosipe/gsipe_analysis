# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:23:02 2023

@author: gsipe
"""

import numpy as np
from scipy.spatial import distance

def get_pw_corr(roi_data, partial_data=None):
    """
    Computes the average pairwise correlation between all neurons across a window.

    Args:
    - roi_data: array containing the ROI data sorted as neurons x time
    - partial_data: optional array containing partial data sorted as neurons x time

    Returns:
    - pw_corr: pairwise correlation matrix
    - pw_corr_ind: pairwise correlation matrix sorted by average correlation
    - ind_pw: indices of pairwise correlation matrix sorted by average correlation
    - avg_pw_corr: average pairwise correlation

    """

    if partial_data is None:
        roi_num = roi_data.shape[0]
        roi_time = roi_data.shape[1]

        roi_norm = np.interp(roi_data, (roi_data.min(), roi_data.max()), (0, 1))

        pw_corr = np.corrcoef(roi_norm, rowvar=False)

        col_mean = np.mean(pw_corr, axis=1)
        ind_pw = np.argsort(col_mean)
        pw_corr_ind = pw_corr[ind_pw][:, ind_pw]

        top_vals = pw_corr[np.triu_indices(roi_num, k=1)]

        avg_pw_corr = np.std(top_vals)

    else:
        roi_num = roi_data.shape[0]
        roi_time = roi_data.shape[1]

        roi_norm = np.interp(roi_data, (roi_data.min(), roi_data.max()), (0, 1))
        partial_norm = np.interp(partial_data, (partial_data.min(), partial_data.max()), (0, 1))

        pw_corr = distance.pdist(np.vstack((roi_norm, partial_norm)).T, metric='correlation')
        pw_corr = distance.squareform(pw_corr)

        col_mean = np.mean(pw_corr, axis=1)
        ind_pw = np.argsort(col_mean)
        pw_corr_ind = pw_corr[ind_pw][:, ind_pw]

        top_vals = pw_corr[np.triu_indices(roi_num, k=1)]

        avg_pw_corr = np.std(top_vals)

    return pw_corr, pw_corr_ind, ind_pw, avg_pw_corr
