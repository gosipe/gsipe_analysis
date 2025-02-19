# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:46:42 2023

@author: Graybird
"""

import numpy as np
from scipy.signal import find_peaks, convolve

def find_wheel(wheel_data, freq_wheel, min_height, min_dur, min_int):
    wheel_event = wheel_data >= min_height  # find values above min_height
    event_thresh = convolve(wheel_event, np.ones(min_dur * freq_wheel), mode='same')  # Take a moving average forward in time over min_dur
    event_thresh = event_thresh >= min_dur  # Values of min_dur indicate a period of at least min_dur above min_height
    event_indices = np.where(event_thresh)[0]  # Find position of events

    if event_indices.size > 0:
        wheel_win = np.zeros((event_indices.size, 2))
        wheel_win[0, 0] = event_indices[0]  # first event index is the start of the first event
        event_diff_indices = np.where(np.diff(event_indices) >= (min_int + min_dur) * freq_wheel)[0]
        wheel_win[1:, 0] = event_indices[event_diff_indices + 1]  # find distance between events => min_int + min_dur
        wheel_win[:-1, 1] = event_indices[event_diff_indices] + min_dur * freq_wheel
        wheel_win[-1, 1] = event_indices[-1] + min_dur * freq_wheel  # The end of the last event period

        if wheel_win[-1, 1] > len(wheel_data):
            wheel_win[-1, 1] = len(wheel_data)  # If end of last window is beyond the end of recording

        wheel_dur = wheel_win[:, 1] - wheel_win[:, 0]  # duration of windows
        wheel_num = wheel_win.shape[0]  # Number of locomotion events

        wheel_amp = np.zeros(wheel_num)
        wheel_loc = np.zeros(wheel_num)
        for i in range(wheel_num):
            wheel_amp[i] = np.max(wheel_data[int(wheel_win[i, 0]):int(wheel_win[i, 1])])  # Maximum wheel amplitude in each window
            wheel_loc[i] = np.argmax(wheel_data[int(wheel_win[i, 0]):int(wheel_win[i, 1])])

    else:
        wheel_num = 0
        wheel_win = np.array([np.nan])
        wheel_dur = np.array([np.nan])
        wheel_loc = np.array([np.nan])
        wheel_amp = np.array([np.nan])

    return wheel_num, wheel_amp, wheel_loc, wheel_win, wheel_dur
