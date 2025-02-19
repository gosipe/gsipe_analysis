# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:09:24 2023

@author: gsipe
"""

import numpy as np
from scipy.interpolate import interp1d

def get_interpol(f, dur, new_rate):
    """
    Interpolates the data matrix or vector to a new frame rate.

    Args:
    - f: data matrix or vector to interpolate (2D array or 1D array)
    - dur: total duration of the time series in seconds (float)
    - new_rate: the new interpolated frame rate (float)

    Returns:
    - int_f: interpolated data (2D array)
    - x_vec: corresponding time vector in seconds for int_f (1D array)

    """

    print('Interpolating DFF...')

    num_roi = f.shape[0]  # Number of ROIs (in rows)
    old_frame = f.shape[1]  # Number of frames (in columns)
    new_frame = int(new_rate * dur)  # The new time vector from interpolated frame rate

    # Define the old vector and the new vector
    old_vec = np.linspace(0, dur, old_frame)
    new_vec = np.linspace(0, dur, new_frame)
    int_f = np.zeros((num_roi, new_frame))  # Preallocate int_f size

    # Interpolate the old f to int_f based on new_vec
    for i in range(num_roi):
        f_interp = interp1d(old_vec, f[i, :])
        int_f[i, :] = f_interp(new_vec)

    # Create the new time vector in seconds
    x_vec = np.arange(1 / new_rate, (new_frame + 1) / new_rate, 1 / new_rate)

    print('...Done!')

    return int_f, x_vec
