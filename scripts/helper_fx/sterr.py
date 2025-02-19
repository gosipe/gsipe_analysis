# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:28:33 2023

@author: Graybird
"""

import numpy as np

def sterr(data, dim=1):
    summary_stats = {}
    summary_stats['samples'] = data.shape[dim]
    summary_stats['mean'] = np.nanmean(data, axis=dim)
    summary_stats['std'] = np.nanstd(data, axis=dim)
    summary_stats['sterr'] = summary_stats['std'] / np.sqrt(data.shape[dim])
    
    return summary_stats
