# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:18:03 2023

@author: Graybird
"""

import numpy as np
import os

def preproc_wheel(master):
    dir_wheel = master['sess_info']['dir_data'] + '/wheel/'
    wheel_mat = np.load(dir_wheel + '*.npy')  # Assumes .npy format for wheel data
    dur_sess = master['sess_info']['dur_sess']
    freq_wheel = 20

    # Setup // Maintenance

    # Parameters // Preferences
    enc_conv = 0.316  # encoder conversion factor from encoder a.u. units to cm
    x_wheel = np.arange(1 / freq_wheel, dur_sess + 1 / freq_wheel, 1 / freq_wheel)  # time vector data
    smooth_win = 5

    # Import Data
    # Move to file director and load neuro.mat
    os.chdir(dir_wheel)
    wheel = np.load(wheel_mat)
    w = wheel['raw']

    # Find Absolute Values // Interpolate // Smooth
    wheel_abs = np.abs(w)  # Find absolute value
    old_vec = np.linspace(0, dur_sess, len(wheel_abs))
    new_vec = np.linspace(0, dur_sess, freq_wheel * dur_sess)
    wheel_interp = np.interp(new_vec, old_vec, wheel_abs, left=0, right=0)
    wheel_clean = wheel_interp
    wheel_clean[wheel_clean <= 1.5] = 0  # Removing jitter values
    wheel_conv = wheel_clean * enc_conv

    wheel_smooth = np.convolve(wheel_conv, np.ones(smooth_win * freq_wheel) / (smooth_win * freq_wheel), mode='same')
    wheel_zsc = (wheel_smooth - np.mean(wheel_smooth)) / np.std(wheel_smooth)
    wheel_norm = (wheel_smooth - np.min(wheel_smooth)) / (np.max(wheel_smooth) - np.min(wheel_smooth))
    wheel_dt = np.append([0], np.diff(wheel_norm))

    # Find periods of movement

    # Save Data
    wheel_info = {
        'enc_conv': enc_conv,
        'freq': freq_wheel,
        'smooth_win': smooth_win,
        'dir': dir_wheel,
        'fn': wheel_mat
    }
    wheel_data = {
        'smooth': wheel_smooth,
        'dt': wheel_dt,
        'zsc': wheel_zsc,
        'norm': wheel_norm,
        'trace': wheel_conv,
        'raw': w,
        'time': wheel['time']
    }

    wheel = {
        'info': wheel_info,
        'xvec': x_wheel,
        **wheel_data
    }

    return wheel
