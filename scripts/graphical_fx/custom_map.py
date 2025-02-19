# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:16:43 2023

@author: Graybird
"""

import numpy as np
from scipy.interpolate import interp1d

def custom_map(mapType='magenta'):
    if mapType == 'sunburst':
        vec = np.array([0, 15, 25, 35, 55, 75, 90, 100])
        hex = ['#000000', '#212728', '#24303D', '#475D75', '#9B3026', '#C16B3A', '#E8E184', '#E0DDBE']
    elif mapType == 'marine':
        vec = np.array([0, 15, 25, 35, 55, 75, 90, 100])
        hex = ['#1A212C', '#2C3E4B', '#34495E', '#00596E', '#1D7872', '#498179', '#71B095', '#F2F2F2']
    elif mapType == 'aqua':
        vec = np.array([0, 40, 60, 100])
        hex = ['#000000', '#1C4D48', '#358F87', '#FFFFFF']
    elif mapType == 'green':
        vec = np.array([0, 40, 60, 80, 100])
        hex = ['#000000', '#04481B', '#088F35', '#46AB68', '#FFFFFF']
    elif mapType == 'magenta':
        vec = np.array([0, 40, 60, 100])
        hex = ['#000000', '#860557', '#C64597', '#FFFFFF']
    elif mapType == 'teal':
        vec = np.array([0, 40, 60, 100])
        hex = ['#000000', '#076B5E', '#47AB9E', '#FFFFFF']
    elif mapType == 'orange':
        vec = np.array([0, 40, 60, 100])
        hex = ['#000000', '#BF6B0A', '#FFA53D', '#FFFFFF']
    elif mapType == 'red':
        vec = np.array([0, 40, 60, 100])
        hex = ['#000000', '#74191F', '#A24147', '#FFFFFF']
    elif mapType == 'gray':
        vec = np.array([0, 20, 40, 50, 65, 80, 100])
        hex = ['#000000', '#191919', '#333333', '#666666', '#999999', '#CCCCCC', '#FFFFFF']
    elif mapType == 'pink':
        vec = np.array([0, 50, 75, 100])
        hex = ['#000000', '#BF5058', '#FF6B75', '#FFFFFF']
    elif mapType == 'earth':
        vec = np.array([0, 35, 65, 100])
        hex = ['#191919', '#5A503D', '#FFA53D', '#FFFFFF']
    elif mapType == 'midnight':
        vec = np.array([0, 30, 60, 100])
        hex = ['#191919', '#37414D', '#92A1B3', '#FFF47A']
    elif mapType == 'invert':
        vec = np.array([0, 20, 40, 50, 65, 80, 100])
        hex = ['#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333', '#191919', '#000000']
    else:
        vec = np.array([0, 40, 60, 100])
        hex = ['#000000', '#860557', '#C64597', '#FFFFFF']

    raw = np.array([int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in hex) / 255.0
    N = 256
    customMap = np.flip(interp1d(vec, raw, axis=0, bounds_error=False, fill_value=(raw[0], raw[-1]))(np.linspace(100, 0, N)), axis=0)

    return customMap
