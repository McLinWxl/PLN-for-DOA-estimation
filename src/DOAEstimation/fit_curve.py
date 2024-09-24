#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt




x = [0, 0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
y = [0.05, 1, 1.6, 1.9, 2.1, 2.1, 2.1, 2.1, 2.1, 2, 1.9]

def target_func(x, a, b, c, d, e, f):
    return a * np.exp(-b * x) + c * np.exp(-d * x) + e * np.exp(-f * x)

def fit_theta(x):
    return 0.3075626 * np.exp(-5.26656769 * x) + 2.04751651 * np.exp(0.02712636 * x)

def fit_gamma(x):
    return 2.99057958 * np.exp(-0.42763038 * x) - 2.94534908 * np.exp(-4.66312438 * x)

# y_fit = [fit_theta(a) for a in x]
# [2.35507911, 2.2347173394430944, 2.1659267568932585, 2.127599112051209, 2.1072683935802723, 2.0975724900165047, 2.0941635312502123, 2.0944737861370357, 2.096986831115797, 2.1008071198096103, 2.105405781117466]

y_fit = [fit_gamma(a) for a in x]
# [0.04509569999999963, 1.0176126387063578, 1.5862698845821404, 1.9033149916473495, 2.0641819954090925, 2.1286682094073788, 2.134218540924307, 2.1042587699375392, 2.053422992080647, 1.9908327406076174, 1.9221539093455777]

# para, cov = opt.curve_fit(target_func, xdata=x, ydata=y, maxfev=500000)

# print(para)

# y_fit = [target_func(a, *para) for a in x]

plt.style.use(['science', 'ieee', 'grid'])
print(y_fit)
plt.plot(y, label='Trained weights')
plt.plot(y_fit, label='Fitted weights')
# plt.plot(fixed, label='Fitted weights (fixed)')
plt.legend()
plt.show()


