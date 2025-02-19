# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:28:47 2023

@author: Graybird
"""

import matplotlib.pyplot as plt
import custom_color
import make_figure
import custom_axes

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
