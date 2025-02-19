# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:06:50 2023

@author: Graybird
"""

import numpy as np
from scipy.interpolate import interp1d

def fill_outliers(data, method, outlier_locations):
    filled_data = np.copy(data)
    outlier_indices = np.where(outlier_locations)[0]
    
    for index in outlier_indices:
        # Find the nearest non-outlier indices before and after the current outlier index
        before_index = np.max(np.where(~outlier_locations[:index])[0])
        after_index = np.min(np.where(~outlier_locations[index+1:])[0]) + index + 1
        
        # Interpolate the data using the non-outlier indices
        f = interp1d([before_index, after_index], [data[before_index], data[after_index]], kind=method)
        filled_data[index] = f(index)
    
    return filled_data

