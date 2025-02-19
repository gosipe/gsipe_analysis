# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:21:34 2023

@author: gsipe
"""

import numpy as np

def get_osi(tune_curve):
    """
    Computes orientation selectivity index (OSI), direction selectivity index (DSI),
    preferred orientation (PO), and preferred direction (PD) from a tuning curve.

    Args:
    - tune_curve: array containing the tuning curve

    Returns:
    - osi: orientation selectivity index
    - dsi: direction selectivity index
    - po: preferred orientation
    - pd: preferred direction

    """

    dir_num = len(tune_curve)

    angle_vec = np.arange(0, 360, 360/dir_num)
    angle_rad = np.deg2rad(angle_vec)

    tc_norm = np.interp(tune_curve, (tune_curve.min(), tune_curve.max()), (0, 1))

    osi = np.abs(np.sum(tc_norm * np.exp(2j * angle_rad)) / np.sum(tc_norm))
    pref_ori = 0.5 * np.angle(np.sum(tc_norm * np.exp(2j * angle_rad)) / np.sum(tc_norm))
    pref_ori = np.rad2deg(pref_ori)

    if pref_ori < 0:
        po = pref_ori + 360
    else:
        po = pref_ori

    dsi = np.abs(np.sum(tc_norm * np.exp(1j * angle_rad)) / np.sum(tc_norm))
    pref_dir = np.angle(np.sum(tc_norm * np.exp(1j * angle_rad)) / np.sum(tc_norm))
    pref_dir = np.rad2deg(pref_dir)

    if pref_dir < 0:
        pd = pref_dir + 360
    else:
        pd = pref_dir

    return osi, dsi, po, pd
