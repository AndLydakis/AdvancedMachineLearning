# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

data = {0: 94.4202,
        1: 85.2373,
        2: 85.8403,
        3: 95.1699,
        4: 81.6528,
        5: 112.966,
        6: 126.015,
        7: 112.671,
        8: 112.643,
        9: 117.437,
        10: 122.161,
        11: 129.264,
        12: 129.705,
        13: 132.546,
        14: 140.269,
        15: 129.878,
        16: 145.896,
        17: 135.23,
        18: 143.732,
        19: 152.035,
        20: 155.396,
        21: 161.172,
        22: 163.666,
        23: 168.873,
        24: 174.94,
        25: 175.308,
        26: 176.143,
        27: 180.292,
        28: 193.266,
        29: 188.549,
        30: 160.153,
        31: 182.635,
        32: 191.628,
        33: 185.812,
        34: 207.821,
        35: 201.36,
        36: 214.379,
        37: 206.205,
        38: 216.187,
        39: 209.672,
        40: 224.692,
        41: 225.556,
        42: 221.654,
        43: 206.715,
        44: 234.158,
        45: 224.737,
        46: 243.967,
        47: 234.546,
        48: 235.39,
        49: 242.918,
        50: 252.5,
        51: 246.885,
        52: 253.141,
        53: 253.051,
        54: 255.129,
        55: 259.826,
        56: 268.333,
        57: 262.223,
        58: 257.521,
        59: 268.022,
        60: 267.412,
        61: 278.415,
        62: 272.703,
        63: 272.959,
        64: 278.84,
        65: 252.245,
        66: 253.105}

x = list(data.keys())
x_squared = np.square(x)
x_cubed = x_squared * x
y = list(data.values())
y = np.array(y)
range_ = np.arange(len(y))
print("LINEAR FEATURES")
A = np.column_stack((np.ones(67), np.arange(67)))
beta = np.linalg.solve((A.T @ A), A.T @ y)
y_approx = list(A @ beta)
plt.plot(range_, y_approx)
plt.plot(range_, y)
plt.show()
plt.cla();

print("POLYNOMIAL DEGREE 2 FEATURES")
A = np.column_stack((x_squared, np.column_stack((np.ones(67), np.arange(67)))))
beta = np.linalg.solve((A.T @ A), A.T @ y)
y_approx = list(A @ beta)
plt.plot(range_, y_approx)
plt.plot(range_, y)
plt.show()
plt.cla();

print("POLYNOMIAL DEGREE 3 FEATURES")
A = np.column_stack((np.ones(67), np.arange(67)))
A = np.column_stack((A, x_squared))
A = np.column_stack((A, x_cubed))
beta = np.linalg.solve((A.T @ A), A.T @ y)
y_approx = list(A @ beta)
plt.plot(range_, y_approx)
plt.plot(range_, y)
plt.show()

print("THREE GROUPS")
x1_ = np.zeros(67)
x2_ = np.zeros(67)
x3_ = np.zeros(67)
x1_[0:23] = 1
x1_[23:45] = 0
x1_[45:67] = 0
x2_[0:23] = 0
x2_[23:45] = 1
x2_[45:67] = 0
x3_[0:23] = 0
x3_[23:45] = 0
x3_[45:67] = 1
# A = np.column_stack((np.ones(67), x1_))
# A = np.column_stack((A, x2_))
# A = np.column_stack((A, x3_))
A = np.column_stack((x1_, x2_))
A = np.column_stack((x2_, x3_))
beta = np.linalg.solve((A.T @ A), A.T @ y)
y_approx = list(A @ beta)
plt.plot(range_, y_approx)
plt.plot(range_, y)
plt.show()

print("TILE SIZE: 8, OFFSET: 7")
tile_size = 8
tile_offset = 7
start = 0
A = np.ones(67)
for o in range(tile_size):
    start = 0
    x_ = np.zeros(67)
    i = o * tile_offset
    while (i < 67):
        while (start <= tile_size and i < 67):
            x_[i] = 1
            start += 1
            i += 1
        while (start >= 0 and i < 67):
            x_[i] = 0
            start -= 1
            i += 1
    A = np.column_stack((A, x_))

beta = np.linalg.solve((A.T @ A), A.T @ y)
y_approx = list(A @ beta)
plt.plot(range_, y_approx)
plt.plot(range_, y)
plt.show()

knots = x[0::10]

t = range(67)
ipl_t = np.linspace(0.0, 67 - 1, 100)
y_tup = si.splrep(t, y, k=3)
y_list = list(y_tup)
yl = y.tolist()
y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]
y_i = si.splev(ipl_t, y_list)

plt.plot(t, y, 'orange')
plt.plot(ipl_t, y_i, 'cornflowerblue')
plt.xlim([0.0, max(t)])
plt.title('Splined y(t)')
plt.show()
