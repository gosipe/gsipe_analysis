# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:30:17 2023

@author: Graybird
"""

import numpy as np
from scipy.optimize import curve_fit

def vm_fxn(x, b, a1, a2, k, pref_rad):
    return b + a1 * np.exp(k * np.cos(x - pref_rad)) + a2 * np.exp(-k * np.cos(x - pref_rad))

def vm_fit(observed, pref_deg, theta_deg=None):
    if theta_deg is None:
        theta_deg = np.arange(0, 360, 360 / len(observed))
    theta_rad = np.deg2rad(theta_deg)

    min_resp = np.min(observed)
    if min_resp < 0:
        observed = observed + np.abs(min_resp)

    coeff_init = [np.mean(observed), np.max(observed), (np.max(observed) + np.mean(observed)) / 2, 7, np.deg2rad(pref_deg)]
    lower_bound = [0, 0, 0, 0, 0]
    upper_bound = [np.mean(observed), np.max(observed), np.max(observed), np.inf, 2 * np.pi]

    try:
        CoeffSet, _ = curve_fit(vm_fxn, theta_rad, observed, p0=coeff_init, bounds=(lower_bound, upper_bound))
        residuals = observed - vm_fxn(theta_rad, *CoeffSet)
        resnorm = np.linalg.norm(residuals)**2
        GOF = 1 - (resnorm / np.linalg.norm(observed - np.mean(observed))**2)
    except RuntimeError:
        CoeffSet = np.zeros_like(coeff_init)
        GOF = np.nan

    return CoeffSet, GOF
