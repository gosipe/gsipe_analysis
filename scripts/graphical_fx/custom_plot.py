# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:19:59 2023

@author: Graybird
"""

import matplotlib.pyplot as plt

def custom_plot(xvec, ydata, color=None, width=1, style='-'):
    if len(xvec) != len(ydata):
        raise ValueError("Cannot plot data because x and y vectors are different lengths")

    def_color = 'k'
    def_width = 1
    def_style = '-'

    if color is None:
        color = def_color

    if width is None:
        width = def_width

    if style is None:
        style = def_style

    p = plt.plot(xvec, ydata, linewidth=width, color=color, linestyle=style)
    plt.hold(True)

    return p
