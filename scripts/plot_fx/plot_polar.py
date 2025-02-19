# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:32:48 2023

@author: Graybird
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear
import custom_color

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
