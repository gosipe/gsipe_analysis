# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:54:31 2023

@author: Graybird
"""

import os
from scipy.signal import find_peaks, medfilt, filtfilt, butter
from scipy.stats import robust_fit
from scipy.signal import find_peaks
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import filtfilt, butter, medfilt
from scipy.stats import zscore, normalize
import numpy as np
import xml.etree.ElementTree as ET



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

def preproc_master(fn):
    dir_xml = os.path.dirname(fn)
    fn_xml = os.path.basename(fn)
    analysis_dir = os.path.join(dir_xml, 'analysis')
    if not os.path.isdir(analysis_dir):
        os.makedirs(analysis_dir)
    plots_dir = os.path.join(dir_xml, 'plots')
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)
    os.chdir(dir_xml)

    # Read XML file and save session info
    sess_info = read_xml(fn_xml)
    fn_new = os.path.splitext(fn_xml)[0]
    anID = fn_new.split('_an')[1].split('_')[0]
    master = {
        'sess_info': sess_info,
        'sess_info.an': anID,
        'sess_info.dir_data': dir_xml
    }
    if 'opto' in fn_xml:
        if 'spon' in fn_xml:
            master['sess_info.type'] = 'spon'
            master['sess_info.grat'] = {
                'grat_on': 2,
                'grat_off': 3,
                'grat_num': 8,
                'trial_num': 32
            }
        elif 'mov' in fn_xml:
            master['sess_info.type'] = 'mov'
            master['sess_info.mov'] = {
                'mov_on': 14,
                'mov_off': 3,
                'mov_num': 7,
                'mov_dur': 2,
                'trial_num': 64
            }
        elif 'grat' in fn_xml:
            master['sess_info.type'] = 'grat'
            master['sess_info.grat'] = {
                'grat_on': 2,
                'grat_off': 3,
                'grat_num': 8,
                'trial_num': 32
            }
        else:
            master['sess_info.type'] = 'spon'
            master['sess_info.grat'] = {
                'grat_on': 2,
                'grat_off': 3,
                'grat_num': 8,
                'trial_num': 32
            }
    else:
        if 'spon' in fn_xml:
            master['sess_info.type'] = 'spon'
            master['sess_info.grat'] = {
                'grat_on': 2,
                'grat_off': 3,
                'grat_num': 8,
                'trial_num': 16
            }
        elif 'mov' in fn_xml:
            master['sess_info.type'] = 'mov'
            master['sess_info.mov'] = {
                'mov_on': 14,
                'mov_off': 3,
                'mov_num': 7,
                'mov_dur': 2,
                'trial_num': 32
            }
        elif 'grat' in fn_xml:
            master['sess_info.type'] = 'grat'
            master['sess_info.grat'] = {
                'grat_on': 2,
                'grat_off': 3,
                'grat_num': 8,
                'trial_num': 16
            }
        else:
            master['sess_info.type'] = 'spon'
            master['sess_info.grat'] = {
                'grat_on': 2,
                'grat_off': 3,
                'grat_num': 8,
                'trial_num': 16
            }

    # Save masterfile
    np.save(os.path.join(analysis_dir, fn_new + '_master.npy'), master)

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
