# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:28:47 2023

@author: Graybird
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import vonmises
from scipy.optimize import curve_fit
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear


def plot_events(data, freq_samp, event_loc):
    # Setup colors and find mean, range, and parameters
    # Open custom colors
    c = custom_color()

    # Calculate the range and mean of the grid data
    min_f = int(min(data))
    max_f = int(max(data))

    dur_plot = len(data) / freq_samp
    # Create time vector in seconds
    xvec = [i / freq_samp for i in range(1, int(dur_plot * freq_samp) + 1)]

    # Create figure with two subplots
    make_figure(14, 8)
    plt.plot(xvec, data, linewidth=1, color=c.gray1)
    plt.ylim([-abs(min_f), max_f])
    plt.hold(True)
    for e in range(event_loc.shape[0]):
        plt.plot(xvec[event_loc[e]], data[event_loc[e]], marker='o', markersize=10,
                 markeredgecolor=c.orange3, linewidth=1)
    plt.xlim([xvec[0], xvec[-1]])
    plt.xlabel('Time(s)')
    custom_axes()
    plt.show()

def plot_fft(data):
    X = data  # put your filtered trace variable here
    Y = np.fft.fft(X)
    L = len(X)
    P2 = np.abs(Y / L)
    P1 = P2[0:L // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]
    f = 20 * (np.arange(0, L // 2 + 1) / L)
    plt.plot(f, P1)

    plt.xlabel('f (Hz)')
    plt.ylabel('|P1(f)|')
    plt.ylim([-np.inf, 0.05])
    plt.box(False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show(block=False)

def plot_polar(tc, num_dir, type='global'):
    """Generate a polar plot for visually tuned responses"""
    c = custom_color()

    x_rad = np.linspace(0, 2*np.pi, num_dir + 1)[:-1]  # create a vector of radians divided by number of directions measured
    if type == 'raw':
        final_tc = np.hstack((tc, tc[0]))  # wrap first and last value to create a continues circular response
        
    elif type == 'zeromin':
        tc = np.hstack((tc, tc[0]))  # wrap first and last value to create a continues circular response
        tc[tc < 0] = 0
        final_tc = tc
        
    elif type == 'scaled':
        tc = np.hstack((tc, tc[0]))  # wrap first and last value to create a continues circular response
        final_tc = (tc - np.min(tc)) / (np.max(tc) - np.min(tc))  # normalizes DFF responses from [0 1]
        
    else:
        tc = np.hstack((tc, tc[0]))  # wrap first and last value to create a continues circular response
        final_tc = tc / np.sum(tc)  # normalizes DFF responses so the sum is 1

    fig = plt.figure(figsize=(6, 6))
    grid_helper = GridHelperCurveLinear((lambda x: x, lambda x: x))
    ax = fig.add_subplot(1, 1, 1, projection='polar', grid_helper=grid_helper)
    ax.set_rgrids([])
    ax.set_thetagrids(np.linspace(0, 360, num_dir + 1)[:-1], frac=1.05, fontsize=14, fontweight='bold')
    ax.set_theta_zero_location('N')
    ax.tick_params(axis='both', which='major', pad=8, length=0)
    ax.set_title('Tuned Responses', fontsize=16, fontweight='normal')
    
    theta = x_rad - np.pi / 2  # shift the polar plot to have the first direction be "up"
    r = final_tc
    cax = ax.plot(theta, r, linewidth=2, color=c.magenta)
    
    return cax, final_tc

def plot_pop(dff, sess_dur):
    """Custom plot of astrocyte grid data"""
    c = custom_color()

    # Calculate the range and mean of the grid data
    mean_dff = np.mean(dff, axis=0)
    min_f = np.floor(np.min(mean_dff))
    max_f = np.ceil(np.max(mean_dff))

    # Calculate the frame rate
    frame_rate = dff.shape[1] / sess_dur
    t_vec = np.arange(1 / frame_rate, dff.shape[1] / frame_rate + 1 / frame_rate, 1 / frame_rate)

    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # Make Grid Raster Heatmap
    axs[0].imshow(dff, vmin=-abs(min_f), vmax=max_f, cmap='magma', aspect='auto')
    axs[0].set_xticks([])
    axs[0].set_title('ROI Raster')
    axs[0].set_ylabel('ROI #')
    axs[0].axis('tight')
    axs[0].grid(False)

    # Create line plot of population average
    axs[1].plot(t_vec, np.nanmean(dff, axis=0), linewidth=1.5, color=c.magenta)
    axs[1].set_ylim(-abs(min_f), max_f)
    axs[1].set_xlim(t_vec[0], t_vec[-1])
    axs[1].set_ylabel('Average DF/F')
    axs[1].set_xlabel('Time(s)')
    axs[1].grid(True)

    plt.tight_layout()
    return fig

def plot_tc(tc_mean, tc_sterr, style, vm_flag, shift_flag):
    # Function to generate a tuning curve plot for visually tuned responses
    c = custom_color()
    num_grat = len(tc_mean)
    angle_inc = 360 / num_grat
    angle_vec = np.arange(0, 360, angle_inc)
    max_idx = np.argmax(tc_mean)
    pref_grat = angle_vec[max_idx]

    if shift_flag:
        angle_vec = np.arange(-180, 180, angle_inc)
        tc_mean = np.roll(tc_mean, num_grat // 2 - 1)
        tc_mean = np.concatenate((tc_mean[-1:], tc_mean))
        tc_sterr = np.roll(tc_sterr, num_grat // 2 - 1)
        tc_sterr = np.concatenate((tc_sterr[-1:], tc_sterr))

    if vm_flag:
        vm_angles = np.linspace(angle_vec[0], angle_vec[-1], 20000)
        vm_rad = np.deg2rad(vm_angles)
        coeff_set, good_fit = vm_fit(tc_mean, pref_grat, angle_vec)
        vm_plot = vm_fxn(coeff_set, vm_rad)

    x_label = [str(ang) for ang in angle_vec]

    if style == 'line':
        plt.errorbar(angle_vec, tc_mean, tc_sterr, fmt='.', markersize=20, linewidth=1, color=c.magenta)
        if vm_flag:
            plt.plot(vm_angles, vm_plot, linewidth=1, color=c.black)
        plt.xlabel('Angle (o)')
        plt.ylabel('Response')
        plt.legend(['TC', 'VM'])
        plt.title('Grating Tuning Curve')
        custom_axes()
        plt.axis('square')
        plt.xlim([-180, 180])
        plt.xticks(angle_vec, x_label)

    elif style == 'shade':
        std_shade(angle_vec, tc_mean, tc_sterr, 0.25, c.magenta)
        if vm_flag:
            plt.plot(vm_angles, vm_plot, linewidth=1, color=c.black)
        plt.xlabel('Angle (o)')
        plt.ylabel('Response')
        plt.legend(['TC', 'VMFit'])
        plt.title('Grating Tuning Curve')
        custom_axes()
        plt.axis('square')

    plt.show()
