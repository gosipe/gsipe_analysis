# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:10:26 2023

@author: Graybird
"""

import matplotlib.pyplot as plt
import custom_color
def custom_box(d, color='black'):
    c = custom_color()

    if color == 'black':
        d['boxprops'] = {'facecolor': c.black, 'alpha': 0.2}
        d['whiskerprops'] = {'color': c.black, 'linestyle': '-'}
        d['boxwidth'] = 0.33
        d['capprops'] = {'color': c.black}
        d['medianprops'] = {'color': c.black}
        d['flierprops'] = {'color': c.black}
        d['widths'] = 0.5
    elif color == 'green':
        d['boxprops'] = {'facecolor': c.green5, 'alpha': 0.5}
        d['whiskerprops'] = {'color': c.green5, 'linestyle': '-'}
        d['boxwidth'] = 0.33
        d['capprops'] = {'color': c.green5}
        d['medianprops'] = {'color': c.green5}
        d['flierprops'] = {'color': c.green5}
        d['widths'] = 0.5

    return d
