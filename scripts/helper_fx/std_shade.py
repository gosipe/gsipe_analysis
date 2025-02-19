# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:24:24 2023

@author: Graybird
"""

import matplotlib.pyplot as plt
import numpy as np

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
