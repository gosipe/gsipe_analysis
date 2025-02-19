# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:52:58 2023

@author: Graybird
"""

import numpy as np
import matplotlib.pyplot as plt
import custom_color
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
