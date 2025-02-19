# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:21:31 2023

@author: Graybird
"""

import matplotlib.pyplot as plt

def make_figure(w=8.5, h=5.5):
    fig_hand = plt.figure(figsize=(w, h))
    left_color = [0, 0, 0]
    right_color = [0, 0, 0]
    fig_hand.set_facecolor('w')
    fig_hand.set_default('axes.prop_cycle', plt.cycler(color=[left_color, right_color]))
    plt.axis('scaled')
    
    return fig_hand
