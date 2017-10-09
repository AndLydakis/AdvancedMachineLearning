# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

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
A = np.column_stack((np.ones(67), np.arange(67)))
beta = np.linalg.solve((A.T @ A), A.T @ y)
y_approx = list(A @ beta)
plt.plot(range_, y_approx)
plt.plot(range_, y)
plt.show()
plt.cla();

print("LINEAR FEATURES")
A = np.column_stack((x_squared, np.column_stack((np.ones(67), np.arange(67)))))
beta = np.linalg.solve((A.T @ A), A.T @ y)
y_approx = list(A @ beta)
plt.plot(range_, y_approx)
plt.plot(range_, y)
plt.show()
plt.cla();

print("POLYNOMIAL FEATURES")
A = np.column_stack((x_cubed,
                     np.column_stack((x_squared, np.column_stack((np.ones(67), np.arange(67)))))))
beta = np.linalg.solve((A.T @ A), A.T @ y)
y_approx = list(A @ beta)
plt.plot(range_, y_approx)
plt.plot(range_, y)
plt.show()

print("THREE GROUPS")
avg1 = sum(y[0:23]) / 22
avg3 = sum(y[23:45]) / 22
avg2 = sum(y[45:67]) / 22
x_ = np.zeros(67)
x_[0:23] = 1
x_[23:45] = 2
x_[45:67] = 3
A = np.column_stack((np.ones(67), x_))
beta = np.linalg.solve((A.T @ A), A.T @ y)
y_approx = list(A @ beta)
plt.plot(range_, y_approx)
plt.plot(range_, y)
plt.show()

print("TILE SIZE: 4")
tile_size = 4
c = 0
sum_ = 0
x_ = np.zeros(67)
for i in range(67):
    sum_ += x[i]
    c += 1
    if c == tile_size:
        x_[i - tile_size + 1:i + 1] = sum_ / tile_size
        c = 0
        sum_ = 0
x_[i - c + 1:67] = sum_ / c
A = np.column_stack((np.ones(67), x_))
beta = np.linalg.solve((A.T @ A), A.T @ y)
y_approx = list(A @ beta)
plt.plot(range_, y_approx)
plt.plot(range_, y)
plt.show()

print("TILE SIZE: 8")
tile_size = 8
c = 0
sum_ = 0
x_ = np.zeros(67)
for i in range(67):
    sum_ += x[i]
    c += 1
    if c == tile_size:
        x_[i - tile_size + 1:i + 1] = sum_ / tile_size
        c = 0
        sum_ = 0
x_[i - c + 1:67] = sum_ / c
A = np.column_stack((np.ones(67), x_))
beta = np.linalg.solve((A.T @ A), A.T @ y)
y_approx = list(A @ beta)
plt.plot(range_, y_approx)
plt.plot(range_, y)
plt.show()

print("TILE SIZE: 16")
tile_size = 16
c = 0
sum_ = 0
x_ = np.zeros(67)
for i in range(67):
    sum_ += x[i]
    c += 1
    if c == tile_size:
        x_[i - tile_size + 1:i + 1] = sum_ / tile_size
        c = 0
        sum_ = 0
x_[i - c + 1:67] = sum_ / c
A = np.column_stack((np.ones(67), x_))
beta = np.linalg.solve((A.T @ A), A.T @ y)
y_approx = list(A @ beta)
plt.plot(range_, y_approx)
plt.plot(range_, y)
plt.show()

print("SPLINE SIZE: 4")
spline_length = 4
i = 0
y_approx_ = np.zeros(0)
while (i < 67):
    x_ = x[i:min(i + spline_length, 67)]
    # print(i, x_)
    A = np.column_stack((np.ones(len(x_)), x_))
    beta = np.linalg.solve((A.T @ A), A.T @ y[i:min(i + spline_length, 67)])
    y_approx = list(A @ beta)
    i += spline_length
    y_approx_ = np.hstack((y_approx_, y_approx))

plt.plot(np.arange(len(y_approx_)), y_approx_)
plt.plot(np.arange(len(y_approx_)), y[0:len(y_approx_)])
plt.show()

print("SPLINE SIZE: 8")
spline_length = 8
i = 0
y_approx_ = np.zeros(0)
while (i < 67):
    x_ = x[i:min(i + spline_length, 67)]
    # print(i, x_)
    A = np.column_stack((np.ones(len(x_)), x_))
    beta = np.linalg.solve((A.T @ A), A.T @ y[i:min(i + spline_length, 67)])
    y_approx = list(A @ beta)
    i += spline_length
    y_approx_ = np.hstack((y_approx_, y_approx))

plt.plot(np.arange(len(y_approx_)), y_approx_)
plt.plot(np.arange(len(y_approx_)), y[0:len(y_approx_)])
plt.show()

print("SPLINE SIZE: 16")
spline_length = 16
i = 0
y_approx_ = np.zeros(0)
while (i < 67):
    x_ = x[i:min(i + spline_length, 67)]
    # print(i, x_)
    A = np.column_stack((np.ones(len(x_)), x_))
    beta = np.linalg.solve((A.T @ A), A.T @ y[i:min(i + spline_length, 67)])
    y_approx = list(A @ beta)
    i += spline_length
    y_approx_ = np.hstack((y_approx_, y_approx))

plt.plot(np.arange(len(y_approx_)), y_approx_)
plt.plot(np.arange(len(y_approx_)), y[0:len(y_approx_)])
plt.show()
