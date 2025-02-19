
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET

def fill_outliers(data, method, outlier_locations):
    filled_data = np.copy(data)
    outlier_indices = np.where(outlier_locations)[0]
    
    for index in outlier_indices:
        # Find the nearest non-outlier indices before and after the current outlier index
        before_index = np.max(np.where(~outlier_locations[:index])[0])
        after_index = np.min(np.where(~outlier_locations[index+1:])[0]) + index + 1
        
        # Interpolate the data using the non-outlier indices
        f = interp1d([before_index, after_index], [data[before_index], data[after_index]], kind=method)
        filled_data[index] = f(index)
    
    return filled_data

def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calculate the intersection point of two lines
    x_intersect = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / ((x1-x2) * (y3-y4) - (y1-y2) * (x3-x4))
    y_intersect = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / ((x1-x2) * (y3-y4) - (y1-y2) * (x3-x4))
    return x_intersect, y_intersect

def read_xml(fn_xml):
    sess_info = {}
    struct_xml = ET.parse(fn_xml).getroot()
    
    sess_info['name'] = fn_xml
    sess_info['date'] = struct_xml.find('PVScan').get('date')
    
    for pv_state_value in struct_xml.findall('.//PVStateValue'):
        key = pv_state_value.get('key')
        value = pv_state_value.get('value')
        
        if key == 'framePeriod':
            sess_info['frame_period'] = float(value)
        elif key == 'dwellTime':
            sess_info['dwellTime'] = float(value)
        elif key == 'linesPerFrame':
            sess_info['lines_per_frame'] = float(value)
        elif key == 'pixelsPerLine':
            sess_info['pixels_per_line'] = float(value)
        elif key == 'objectiveLens':
            sess_info['objective'] = value
        elif key == 'objectiveLensNA':
            sess_info['objective_NA'] = value
        elif key == 'opticalZoom':
            sess_info['optical_zoom'] = float(value)
        elif key == 'rastersPerFrame':
            sess_info['rasters_per_frame'] = float(value)
    
    sequence = struct_xml.find('.//Sequence')
    if sequence is not None and sequence.find('Frame') is not None:
        frames = sequence.findall('Frame')
        sess_info['num_frame'] = len(frames)
        sess_info['frame_time'] = []
        for frame in frames:
            relative_time = float(frame.get('relativeTime'))
            sess_info['frame_time'].append(relative_time)
        
        frame_time1 = float(frames[0].get('absoluteTime'))
        frame_time2 = float(frames[1].get('absoluteTime'))
        sess_info['frame_rate'] = 1 / (frame_time2 - frame_time1)
        last_frame = sess_info['frame_time'][-1]
        sess_info['dur_sess'] = round(last_frame)
    
    elif sequence is not None and sequence[0].find('Frame') is not None:
        frames = sequence[0].findall('Frame')
        sess_info['num_frame'] = len(sequence)
        z_piezo_start = float(struct_xml.find('.//SubindexedValue[@id="20_3_2"]').get('value'))
        z_piezo_second = float(sequence[0].find('.//SubindexedValue[@id="2_3_2"]').get('value'))
        sess_info['zpiezo'] = {
            'zstep': z_piezo_second - z_piezo_start,
            'znum': len(frames)
        }
        sess_info['frame_time'] = []
        for frame in frames:
            relative_time = float(frame.get('relativeTime'))
            sess_info['frame_time'].append(relative_time)
        
        frame_time1 = float(frames[0].get('absoluteTime'))
        frame_time2 = float(sequence[1].find('Frame').get('absoluteTime'))
        sess_info['frame_rate'] = 1 / (frame_time2 - frame_time1)
        last_frame = sess_info['frame_time'][-1]
        sess_info['dur_sess'] = round(last_frame)
    
    return sess_info

def std_shade(xax, amean, astd, alpha, acolor):
    if alpha is None:
        plt.fill_between(xax, amean+astd, amean-astd, color=acolor, linestyle='none')
        acolor = 'k'
    else:
        plt.fill_between(xax, amean+astd, amean-astd, color=acolor, alpha=alpha, linestyle='none')
    
    plt.plot(xax, amean, '-', color=acolor, linewidth=1, markersize=8, markerfacecolor=acolor, markeredgecolor=acolor)
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.show()

def sterr(data, dim=1):
    summary_stats = {}
    summary_stats['samples'] = data.shape[dim]
    summary_stats['mean'] = np.nanmean(data, axis=dim)
    summary_stats['std'] = np.nanstd(data, axis=dim)
    summary_stats['sterr'] = summary_stats['std'] / np.sqrt(data.shape[dim])
    
    return summary_stats

def vm_fxn(x, b, a1, a2, k, pref_rad):
    return b + a1 * np.exp(k * np.cos(x - pref_rad)) + a2 * np.exp(-k * np.cos(x - pref_rad))

def vm_fit(observed, pref_deg, theta_deg=None):
    if theta_deg is None:
        theta_deg = np.arange(0, 360, 360 / len(observed))
    theta_rad = np.deg2rad(theta_deg)

    min_resp = np.min(observed)
    if min_resp < 0:
        observed = observed + np.abs(min_resp)

    coeff_init = [np.mean(observed), np.max(observed), (np.max(observed) + np.mean(observed)) / 2, 7, np.deg2rad(pref_deg)]
    lower_bound = [0, 0, 0, 0, 0]
    upper_bound = [np.mean(observed), np.max(observed), np.max(observed), np.inf, 2 * np.pi]

    try:
        CoeffSet, _ = curve_fit(vm_fxn, theta_rad, observed, p0=coeff_init, bounds=(lower_bound, upper_bound))
        residuals = observed - vm_fxn(theta_rad, *CoeffSet)
        resnorm = np.linalg.norm(residuals)**2
        GOF = 1 - (resnorm / np.linalg.norm(observed - np.mean(observed))**2)
    except RuntimeError:
        CoeffSet = np.zeros_like(coeff_init)
        GOF = np.nan

    return CoeffSet, GOF

