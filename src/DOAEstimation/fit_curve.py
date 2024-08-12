#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

t = [-0.18053345, -0.17243316, -0.17663853, -0.16958929, -0.17017754,
       -0.18197065, -0.16959207, -0.1690475 , -0.177824  , -0.1763189 ,
       -0.17905227, -0.17662498, -0.17270402, -0.16323605, -0.17049725,
       -0.16758798, -0.17243101, -0.16740706, -0.1691505 , -0.16824611,
       -0.16002157, -0.16429888, -0.16434333, -0.16354133, -0.17479465,
       -0.17808053, -0.1735309 , -0.18057317, -0.17829627, -0.18448684,
       -0.17414507, -0.17872746, -0.17935621, -0.17853242, -0.17962647,
       -0.17799058, -0.17480377, -0.18595738, -0.18989318, -0.1924971 ,
       -0.20599245, -0.20556037, -0.22749936, -0.2447727 , -0.26772983,
       -0.27932498, -0.30611612, -0.3199457 , -0.33744165, -0.36386117,
       -0.38406786, -0.39655213, -0.41382327, -0.42474411, -0.4457523 ,
       -0.45355386, -0.4518928 , -0.45739413, -0.46130525, -0.46128138,
       -0.45521018, -0.45981932, -0.44694097, -0.44454323, -0.4360007 ,
       -0.43193633, -0.42130352, -0.41462646, -0.40124626, -0.38246235,
       -0.36707168, -0.36211733, -0.35438625, -0.34539411, -0.35204072,
       -0.35405937, -0.35465178, -0.35616539, -0.35344169, -0.3427969 ]

fit_layers = 50

# save as txt
# with open('weights.txt', 'w') as f:
#     for idd, item in enumerate(t):
#         f.write(str(idd) + '\t' + str(item) + '\n')


t = t[-50:]

def target_func(x, a, b, c):
    return a*

para, cov = opt.curve_fit(target_func, xdata=np.arange(fit_layers), ydata=t, maxfev=500000)

print(para)

y_fit = [target_func(a, *para) for a in np.arange(fit_layers)]

plt.style.use(['science', 'ieee', 'grid'])

plt.plot(t, label='Trained weights')
plt.plot(y_fit, label='Fitted weights')
plt.legend()
plt.show()
