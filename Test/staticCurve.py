#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import matplotlib.pyplot as plt
import numpy as np

SNR_set = [-20, -15, -10, -5, 0, 5, 10, np.inf]
layersProposed_set = [10, 20, 40, 80, 160, 320]
layersIterative_set = [10, 20, 40, 80, 160, 320, 500, 640, 1000, 1280, 1500, 2560]

proposed_NMSE = np.zeros((len(SNR_set), len(layersProposed_set)))
proposed_NMSE[4, :] = [-2.4646280705928802, -5.928625464439392, -5.79640805721283, -3.955305814743042, -10.461779832839966, -13.11316728591919]

ISTA_NMSE = np.zeros((len(SNR_set), len(layersIterative_set)))
ISTA_NMSE[4, :] = [5.744519829750061, 4.632649719715118, 3.400360941886902, 1.939157098531723, 0.38167357444763184, -1.858755350112915, -5.442613959312439, -10.204370021820068, -14.345990419387817]
ISTA15_NMSE = np.zeros((len(SNR_set), len(layersIterative_set)))
ISTA15_NMSE[4, :] = [3.776443898677826, 2.4561329185962677, 0.9740715473890305, -0.8984984457492828, -3.9317935705184937, -8.202711939811707, 11.171270608901978, -12.687160968780518, -15.044164657592773, -16.098709106445312, , -17.533432245254517]


plt.style.use(['science', 'ieee', 'grid'])

plt.figure(dpi=800)
SNR = 0
idx = SNR_set.index(SNR)
plt.plot(layersProposed_set, proposed_NMSE[idx], ls='-', label='Proposed', marker='o', markersize=3, color='darkblue')

plt.plot(layersIterative_set, ISTA_NMSE[idx], ls='-', label='ISTA ($\\alpha=1$)', marker='o', markersize=3, color='darkred')
plt.plot(layersIterative_set, ISTA15_NMSE[idx], ls='-', label='ISTA ($\\alpha=1.5$)', marker='x', markersize=4, color='darkred')
plt.xlabel('Iterations / Layers (k)')
plt.ylabel('NMSE (dB)')
# plt.title('')
plt.legend()
plt.savefig(f'../Test/Figures/testStatic/{SNR}dB.pdf')
plt.show()

