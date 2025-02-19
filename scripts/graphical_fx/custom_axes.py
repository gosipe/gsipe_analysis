# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:09:40 2023

@author: Graybird
"""

import matplotlib.pyplot as plt

def custom_axes(a=None, f=None):
    if a is None:
        a = plt.gca()
    if f is None:
        f = plt.gcf()

    a.tick_params(length=0.02)
    a.set_facecolor('white')
    a.set_linewidth(1.5)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.spines['left'].set_visible(False)
    a.tick_params(direction='in', labelsize=12)
    a.title.set_fontsize(12)
    a.title.set_fontweight('normal')

    # Uncomment the following lines if you want to customize the legend
    # lgd = a.legend()
    # lgd.set_box(False)
    # lgd.get_frame().set_visible(False)
    # lgd.loc = 'lower outside'
    # lgd.orientation = 'horizontal'

    plt.box(False)
