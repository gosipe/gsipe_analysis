# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:31:07 2023

@author: Graybird
"""

import numpy as np
import matplotlib.pyplot as plt

# FFT original and filtered
plt.figure('FFT original and filtered', figsize=(10, 6))

X = d_zsc
Y = np.fft.fft(X)
L = len(X)
P2 = np.abs(Y / L)
P1 = P2[0:L // 2 + 1]
P1[1:-1] = 2 * P1[1:-1]
f = 20 * (np.arange(0, L // 2 + 1) / L)
plt.plot(f, P1)

X = filt_zsc  # put your filtered trace variable here
Y = np.fft.fft(X)
L = len(X)
P2 = np.abs(Y / L)
P1 = P2[0:L // 2 + 1]
P1[1:-1] = 2 * P1[1:-1]
f = 20 * (np.arange(0, L // 2 + 1) / L)
plt.plot(f, P1)

plt.xlabel('f (Hz)')
plt.ylabel('|P1(f)|')
plt.ylim([-np.inf, 0.05])
plt.box(False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
