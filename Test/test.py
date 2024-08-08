import sys
sys.path.append('../src')
import numpy as np

import src.DataGenerator

import matplotlib.pyplot as plt

plt.style.use(['science', 'ieee', 'grid'])

num_iters = 40
x_axis = np.arange(num_iters)
wights_ = [ 6.69332518e-14, -1.46119797e-11,  1.35070284e-09, -6.88957794e-08,
        2.10771018e-06, -3.90486205e-05,  4.15174983e-04, -2.18875963e-03,
        4.76018000e-03, -4.14020957e-03, -6.24989189e-04]
wights_ = wights_[::-1]
curve_ = np.zeros_like(x_axis)
for i in range(len(wights_)):
    curve_ = curve_ + wights_[i] * x_axis ** i
plt.plot(x_axis, curve_)
plt.show()
