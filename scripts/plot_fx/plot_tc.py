# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:55:00 2023

@author: Graybird
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from scipy.optimize import curve_fit
import custom_color
import vm_fit
import vm_fxn
import custom_axes
import std_shade
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
