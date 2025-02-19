# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:08:55 2023

@author: Graybird
"""

import numpy as np

def preproc_suite2p(dir_neuro, method='Fixed'):
    np.warnings.filterwarnings('ignore')  # Suppress warnings
    # Move to directory with Suite2P NPY files
    npy_dir = dir_neuro + '/suite2p/plane0/'
    if method not in ['Fixed', 'Robust']:
        method = 'Fixed'

    # Read in Suite2P outputs
    all_f = np.load(npy_dir + 'F.npy').T  # All ROI fluorescence
    all_np = np.load(npy_dir + 'Fneu.npy').T  # All ROI neuropil
    is_cell = np.load(npy_dir + 'iscell.npy')[:, 0].astype(bool)  # Logical index for ROIs determined to be cells

    # Select only ROIs and neuropils that are cells
    cell_f = all_f[:, is_cell].T  # Cell fluorescence
    cell_np = all_np[:, is_cell].T  # Cell neuropil

    # Calculate number of ROIs and frames
    num_roi = cell_f.shape[0]
    num_frame = cell_f.shape[1]

    # Choose method to subtract neuropil fluorescence from cell
    if method == 'Fixed':
        np_corr_data = cell_f - (0.7 * cell_np)
    elif method == 'Robust':
        np_corr_data = np.zeros((num_roi, num_frame))
        for i in range(num_roi):
            b = np.linalg.lstsq(cell_np[i, :].reshape(-1, 1), cell_f[i, :].reshape(-1, 1), rcond=None)[0]
            np_corr_data[i, :] = cell_f[i, :] - b[0] * cell_np[i, :]

    neuro_raw = np_corr_data

    # Save the data to a .npy file
    np.save('neuro_raw.npy', neuro_raw)
