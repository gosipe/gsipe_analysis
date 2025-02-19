# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:54:31 2023

@author: Graybird
"""

import os
from scipy.signal import find_peaks, medfilt, filtfilt, butter
import numpy as np
import get_interpol
import get_dff

def preproc_astro(master):
    # Parameters / Preferences
    smooth_win = 3
    act_thresh = 0.5  # Threshold for an ROI to be active
    freq_thresh = 10
    peak_dist = 5

    # Import Data
    dir_astro = os.path.join(master['sess_info']['dir_data'], 'astro')
    astro_txt = [f for f in os.listdir(dir_astro) if f.endswith('.txt')][0]
    dur_sess = master['sess_info']['dur_sess']
    freq_astro = int(np.ceil(master['sess_info']['frame_rate']))

    # Move to file directory and load astrocyte ROI .txt file
    os.chdir(dir_astro)
    astro_raw = np.loadtxt(astro_txt).T

    x_astro = np.arange(1 / freq_astro, dur_sess + 1e-6, 1 / freq_astro)  # time vector for astrocyte traces

    astro = {
        'xvec': x_astro,
        'info': {
            'dir': dir_astro,
            'fn': astro_txt,
            'num_roi': astro_raw.shape[0]
        }
    }

    # Interpolate frames and calculate DFF
    num_roi = astro_raw.shape[0]  # number of ROIs assuming rows are cells and columns are time
    astro_new = get_interpol(astro_raw, dur_sess, freq_astro)
    astro_dff = get_dff(astro_new)

    astro['raw'] = astro_raw
    astro['dff'] = astro_dff
    astro['info']['num_roi'] = num_roi

    # Find Active Grid Squares
    peak_num = np.zeros(num_roi)
    for a in range(num_roi):
        peaks, _ = find_peaks(medfilt(astro_dff[a, :], smooth_win), distance=freq_astro * peak_dist, prominence=act_thresh)
        peak_num[a] = len(peaks)

    active_roi = np.where(peak_num >= freq_thresh)[0]
    active_idx = astro_dff[active_roi, :] >= act_thresh
    astro['active'] = {
        'idx': active_idx,
        'fract': np.sum(active_idx, axis=0) / active_roi.shape[0],
        'dff': astro_dff[active_roi, :],
        'avg': np.mean(astro_dff[active_roi, :], axis=0),
        'zsc': np.mean(astro_dff[active_roi, :], axis=0)
    }

    # Filter data and smooth with moving median window
    filt_order = 2  # order of filter to remove pupillary light response
    freq_band = [1, freq_astro / 2]  # frequency range to filter out to remove pupillary light response and noise

    # Create filter using 'butter' bandpass method
    b, a = butter(filt_order, freq_band, fs=freq_astro, btype='bandstop')
    astro_filt = filtfilt(b, a, astro['active']['avg'])  # filter interpolated data
    astro_filt = np.convolve(astro_filt, np.ones(smooth_win * freq_astro) / (smooth_win * freq_astro), mode='same')  # smooth filtered data
    astro_zsc = (astro_filt - np.mean(astro_filt)) / np.std(astro_filt)  # calculate zscore of smoothed data
    astro['filt'] = {
        'dff': astro_filt,
        'zsc': astro_zsc
    }
    astro['info']['freq_astro'] = freq_astro

    # Save variables
    astro['info']['act_thresh'] = act_thresh
    astro['info']['smoothwin'] = smooth_win
    astro['info']['freq_thresh'] = freq_thresh
    astro['info']['peak_dist'] = peak_dist

    return astro
