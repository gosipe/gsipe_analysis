# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:55:13 2023

@author: Graybird
"""

import os
import numpy as np
from scipy.stats import robust_fit
from scipy.signal import find_peaks
import get_interpol
import get_dff

def preproc_neuro(master, method='Fixed'):
    # Move to directory with suite2p NPY files
    dir_neuro = os.path.join(master['sess_info']['dir_data'], 'neuro', 'suite2p', 'plane0')
    os.chdir(dir_neuro)
    dur_sess = master['sess_info']['dur_sess']
    freq_neuro = int(np.ceil(master['sess_info']['frame_rate']))

    # Read in suite2P outputs
    all_f = np.transpose(np.load('F.npy'))  # All ROI fluorescence
    all_np = np.transpose(np.load('Fneu.npy'))  # All ROI neuropil
    is_cell = np.load('iscell.npy')  # Logical index for ROIs determined to be cells

    # Select only ROIs and neuropils that are cells
    cell_f = all_f[:, np.where(is_cell[:, 0])[0]]  # Cell fluorescence
    cell_np = all_np[:, np.where(is_cell[:, 0])[0]]  # Cell neuropil

    # Calculate number of ROIs and frames
    num_roi = cell_f.shape[0]
    num_frame = cell_f.shape[1]

    # Choose method to subtract neuropil fluorescence from cell
    if method == 'Fixed':
        np_corr_data = cell_f - (0.7 * cell_np)
    elif method == 'Robust':
        np_corr_data = np.zeros((num_roi, num_frame))
        for i in range(num_roi):
            b = robust_fit(cell_np[i, :], cell_f[i, :])
            np_corr_data[i, :] = cell_f[i, :] - b[1] * cell_np[i, :]
    else:
        raise ValueError("Invalid method specified.")

    neuro_raw = np_corr_data
    x_neuro = np.arange(1 / freq_neuro, dur_sess + 1e-6, 1 / freq_neuro)  # time vector for neuronal trace

    # Thresholds for determining activity
    action_thresh = 0.5

    # Calculate DFF / Interpolate Frames
    neuro = {
        'info': {
            'freq_neuro': freq_neuro
        },
        'xvec': x_neuro,
        'raw': neuro_raw
    }
    num_roi = neuro_raw.shape[0]  # number of ROIs assuming rows are cells and columns are time
    neuro_new = get_interpol(neuro_raw, dur_sess, freq_neuro)
    neuro_dff = get_dff(neuro_new)
    roi_activity_num = np.zeros(num_roi)

    neuro['new'] = neuro_new
    neuro['dff'] = neuro_dff
    neuro['avg'] = np.mean(neuro_dff, axis=0)
    neuro['roi'] = []

    for roi in range(num_roi):
        curr_roi = neuro_dff[roi, :]
        peaks, _ = find_peaks(curr_roi, height=action_thresh, distance=freq_neuro)
        roi_activity_num[roi] = len(peaks)
        neuro['roi'].append({
            'dff': neuro_dff[roi, :],
            'activity_num': roi_activity_num[roi]
        })

    return neuro
