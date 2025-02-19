# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:56:56 2023

@author: Graybird
"""
import os 
import glob
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import filtfilt, butter, medfilt
from scipy.stats import zscore, normalize
import fill_outliers
import get_dff
import line_intersection

def preproc_pupil(master):
    dur_sess = master['sess_info']['dur_sess']
    dir_pupil = master['sess_info']['dir_data'] + '/pupil'
    pupil_csv = glob.glob(os.path.join(dir_pupil, '*.csv'))[0]
    tstamp_txt = glob.glob(os.path.join(dir_pupil, '*.txt'))[0]
    # Setup parameters and preferences
    conv_factor = 60.8  # conversion factor for pixels to 1 mm based on camera setup measurement
    interp_method = 'cubic'  # method used for filling outliers and interpolating data
    t_stamp_interp_flag = True  # old camera, set flag to 'True'.... new camera, set flag to 'False'
    # Frequency setup
    freq_pupil = 20  # frequency to interpolate pupil trace
    smooth_win = freq_pupil // 2
    x_pupil = np.arange(1 / freq_pupil, dur_sess + 1 / freq_pupil, 1 / freq_pupil)  # x vector for plotting pupil as a time vector

    # Go to directory and import data
    os.chdir(dir_pupil)
    # Import time stamp text file
    with open(tstamp_txt, 'r') as file:
        time_txt = file.readlines()
    t_stamp = [float(i) for i in time_txt[1:]]  # Remove first time stamp which corresponds to first image
    orig_tstamp = t_stamp
    # Import pupil csv file
    raw_csv = np.loadtxt(pupil_csv, delimiter=',', skiprows=1, usecols=[1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23])

    # Pull out the x/y coordinates for each body part
    bp1 = raw_csv[:, 0:2]
    bp2 = raw_csv[:, 2:4]
    bp3 = raw_csv[:, 4:6]
    bp4 = raw_csv[:, 6:8]
    bp5 = raw_csv[:, 8:10]
    bp6 = raw_csv[:, 10:12]
    bp7 = raw_csv[:, 12:14]
    bp8 = raw_csv[:, 14:16]

    # Calculate pupil diameter as horizontal diameter scaled to vertical diameter
    # Find the vertical pupil diameter (bp1 to bp5)
    vert_diam = np.sqrt((bp1[:, 0] - bp5[:, 0])**2 + (bp1[:, 1] - bp5[:, 1])**2)

    # Find the horizontal pupil diameter(bp3 to bp7)
    hor_diam = np.sqrt((bp3[:, 0] - bp7[:, 0])**2 + (bp3[:, 1] - bp7[:, 1])**2)

    # Calculate median ratio to scale horizontal distance to vertical distance
    ratio_diam = np.median(vert_diam / hor_diam)

    # Scale horizontal diameter and convert to mm
    diam_conv = (hor_diam * ratio_diam) / conv_factor
    dt_conv = np.insert(np.diff(diam_conv), 0, 0)
    dt_art_low = np.percentile(dt_conv[dt_conv < 0], 5)
    dt_art_high = np.percentile(dt_conv[dt_conv > 0], 5)
    art_loc = np.where((dt_conv < dt_art_low) | (dt_conv > dt_art_high))[0]
    art_vec = np.zeros_like(diam_conv, dtype=bool)
    art_vec[art_loc] = True
    conv_fix = fill_outliers(diam_conv, interp_method, 'OutlierLocations', art_vec)

    # Interpolate pupil diameter to time stamps using specified interpolation method
    if t_stamp_interp_flag:
        if len(t_stamp) > len(diam_conv):
            t_stamp = t_stamp[:len(diam_conv)]
        if len(t_stamp) < len(diam_conv):
            t_stamp = np.append(t_stamp, t_stamp[-1] + 1 / freq_pupil)
        f = interp1d(t_stamp, conv_fix, kind=interp_method, fill_value="extrapolate")
        pupil_diam = f(x_pupil)
    else:
        frame_vector = np.arange(1 / freq_pupil, len(conv_fix) / freq_pupil + 1 / freq_pupil, 1 / freq_pupil)
        f = interp1d(frame_vector, conv_fix, kind=interp_method, fill_value="extrapolate")
        pupil_diam = f(x_pupil)
    pupil_diam = fill_outliers(pupil_diam, interp_method, 'OutlierLocations', np.isnan(pupil_diam))

    # Filter data and smooth with moving median window
    filt_order = 4  # order of filter to remove pupillary light response
    freq_band = [0.5, 10]  # frequency range to filter out to remove pupillary light response and noise
    # Create filter using 'butter' bandpass method
    b, a = butter(filt_order, freq_band, fs=freq_pupil, btype='bandstop')
    d_filt = filtfilt(b, a, pupil_diam)  # filter interpolated data
    d_zsc = zscore(pupil_diam)  # calculate zscore of diameter
    filt_zsc = zscore(d_filt)  # calculate zscore of filtered diameter
    d_dpp = get_dff(d_filt, 'Method2')
    d_norm = normalize(d_filt, axis=0)
    d_dt = np.insert(np.diff(d_norm), 0, 0)

    # Find center of pupil and calculate x/y coordinates over time
    p_coord = np.zeros((len(diam_conv), 2))  # Set up center coordinate vector matrix

    # For each frame calculate the x/y intersection of 3 lines
    # Lines are bp2-6, bp3-7, and bp4-8
    for frame in range(len(diam_conv)):
        x1, y1 = line_intersection([bp2[frame, 0], bp6[frame, 0]], [bp2[frame, 1], bp6[frame, 1]], [bp3[frame, 0], bp7[frame, 0]], [bp3[frame, 1], bp7[frame, 1]])
        x2, y2 = line_intersection([bp2[frame, 0], bp6[frame, 0]], [bp2[frame, 1], bp6[frame, 1]], [bp4[frame, 0], bp8[frame, 0]], [bp4[frame, 1], bp8[frame, 1]])
        x3, y3 = line_intersection([bp2[frame, 0], bp4[frame, 0]], [bp8[frame, 1], bp4[frame, 1]], [bp4[frame, 0], bp8[frame, 0]], [bp3[frame, 1], bp7[frame, 1]])
        # Average the 3 intersections
        xC = np.nanmean([x1, x2, x3])
        yC = np.nanmean([y1, y2, y3])
        # Store x/y coordinates and deal with NaNs
        if np.isnan(xC[0]) and np.isnan(xC[1]) and np.isnan(xC[2]):
            p_coord[frame, 0] = np.nan
            p_coord[frame, 1] = np.nan
        elif np.isnan(yC[0]) and np.isnan(yC[1]) and np.isnan(yC[2]):
            p_coord[frame, 0] = np.nan
            p_coord[frame, 1] = np.nan
        else:
            p_coord[frame, 0] = xC
            p_coord[frame, 1] = yC

    p_filt_coord = fill_outliers(p_coord, 'makima', 'OutlierLocations', np.isnan(p_coord))

    # Filter the center coordinates and plot
    # Fill outliers in a moving median window defined by sWin
    # p_filt_cord = filloutliers(intrpC, 'linear', 'movmedian', smooth_win*2)

    # Interpolate center coordinates to tStamp list
    if t_stamp_interp_flag:
        f = interp1d(t_stamp, p_filt_coord, axis=0, kind=interp_method, fill_value="extrapolate")
        p_filt_coord = f(x_pupil)
    else:
        f = interp1d(frame_vector, p_filt_coord, axis=0, kind=interp_method, fill_value="extrapolate")
        p_filt_coord = f(x_pupil)
    p_filt_coord = fill_outliers(p_filt_coord, 'previous', 'OutlierLocations', np.isnan(p_filt_coord))

    # Calculate deviation of pupil movements from mean position
    meanXC = np.mean(p_filt_coord[~np.isnan(p_filt_coord[:, 0]), 0], axis=0)
    meanYC = np.mean(p_filt_coord[~np.isnan(p_filt_coord[:, 1]), 1], axis=0)
    p_dist = np.sqrt((p_filt_coord[:, 0] - meanXC) ** 2 + (p_filt_coord[:, 1] - meanYC) ** 2) / conv_factor
    p_smooth = medfilt(p_dist, smooth_win)

    p_zsc = zscore(p_smooth)
    p_norm = normalize(p_smooth, axis=0)
    p_dt = np.insert(np.diff(p_norm), 0, 0)

    # Save data
    pupil = {}
    pupil['xvec'] = x_pupil

    pupil['info'] = {}
    pupil['info']['conv_facto'] = conv_factor
    pupil['info']['freq'] = freq_pupil
    pupil['info']['ratio_diam'] = ratio_diam
    pupil['info']['interp_method'] = interp_method
    pupil['info']['dir_pupil'] = dir_pupil
    pupil['info']['fn_pupil'] = pupil_csv.name
    pupil['info']['fn_time'] = tstamp_txt.name
    pupil['info']['t_stamp'] = t_stamp
    pupil['info']['orig_tstamp'] = orig_tstamp
    pupil['info']['smooth_win'] = smooth_win
    pupil['info']['t_stamp_interp_flag'] = t_stamp_interp_flag

    pupil['diam'] = {}
    pupil['diam']['filt'] = d_filt
    pupil['diam']['filt_zsc'] = filt_zsc
    pupil['diam']['dpp'] = d_dpp
    pupil['diam']['zscore'] = d_zsc
    pupil['diam']['norm'] = d_norm
    pupil['diam']['dt'] = d_dt

    pupil['coord'] = {}
    pupil['coord']['raw'] = p_coord
    pupil['coord']['filt'] = p_filt_coord
    pupil['coord']['dist'] = p_dist
    pupil['coord']['smooth'] = p_smooth
    pupil['coord']['zscore'] = p_zsc
    pupil['coord']['norm'] = p_norm
    pupil['coord']['dt'] = p_dt

    return pupil
