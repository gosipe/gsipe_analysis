# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:24:42 2023

@author: gsipe
"""

import numpy as np

def get_quartiles(data_vec):
    """
    Computes quartiles and indices of data vector.

    Args:
    - data_vec: 1D array of data

    Returns:
    - quartiles: dictionary containing quartile values and indices

    """

    data_vec = np.squeeze(data_vec)
    if data_vec.ndim == 1:
        data_vec = np.transpose(data_vec)

    q1 = np.percentile(data_vec, 25)
    q2 = np.percentile(data_vec, 50)
    q3 = np.percentile(data_vec, 75)
    q4 = np.percentile(data_vec, 100)

    quartiles = {}
    quartiles['vals'] = np.array([q1, q2, q3, q4])

    idx_q1 = np.where(data_vec < q1)[0]
    idx_q2 = np.where((data_vec >= q1) & (data_vec < q2))[0]
    idx_q3 = np.where((data_vec >= q2) & (data_vec < q3))[0]
    idx_q4 = np.where(data_vec >= q3)[0]

    quartiles['idx_q1'] = idx_q1
    quartiles['idx_q2'] = idx_q2
    quartiles['idx_q3'] = idx_q3
    quartiles['idx_q4'] = idx_q4

    return quartiles
