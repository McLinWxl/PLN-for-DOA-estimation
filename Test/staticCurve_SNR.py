#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import matplotlib.pyplot as plt
import numpy as np

layersProposed_set = [-20, -15, -10, -5, 0, 5, 10]

proposed_NMSE = [0.13285637833178043, 0.009900444420054555, -1.1975890398025513, -5.370664000511169, -13.8423752784729, -22.09010362625122, -23.266282081604004]
ISTA_1 = [0.3193768113851547, 0.21494576707482338, -0.7611924409866333, -4.290632903575897, -8.926278948783875, -12.159531116485596, -13.224071264266968]
# ISTA_32 =
ISTA_2 = [0.6593075394630432, 0.3965235874056816, -0.9516724199056625, -4.5756995677948, -8.47753643989563, -10.695067644119263, -10.874572992324829]
# ISTA_12 =
ISTA_14 = [0.13840829953551292, 0.11576320976018906, -0.5162682011723518, -3.587430417537689, -9.14508581161499, -15.697109699249268, -23.15173864364624]

pro_DOA = [9.42838504, 7.14881498, 2.99629401, 0.39440532, 0.0745356, 0.0745356, 0.0745356]
ISTA_1_DOA = [9.47628619, 7.52809552, 3.51188458, 0.70710678, 0.30731815, 0.19720266, 0.19720266]
ISTA_2_DOA = [9.46983515, 7.36998417, 4.00208279, 0.61913919, 0.28867513, 0.2236068, 0.23570226]
ISTA_14_DOA = [9.51694395, 8.93743687, 4.94750218, 1.8944363, 0.34156503, 0.12909944, 0.04]

plt.style.use(['science', 'ieee', 'grid'])
plt.figure(dpi=800)

plt.plot(layersProposed_set, ISTA_2, ls='-', label='ISTA ($\\alpha=2$; 32 Iterations)', marker='x', markersize=3, color='darkred')
# plt.plot(layersProposed_set, ISTA_32, ls='-', label='ISTA ($\\alpha=1.5$)', marker='v', markersize=3, color='darkred')
plt.plot(layersProposed_set, ISTA_1, ls='-', label='ISTA ($\\alpha=1$; 96 Iterations)', marker='o', markersize=3, color='darkred')
# plt.plot(layersProposed_set, ISTA_12, ls='-', label='ISTA ($\\alpha=0.5$)', marker='^', markersize=3, color='darkred')
plt.plot(layersProposed_set, ISTA_14, ls='-', label='ISTA ($\\alpha=0.25$; 320 Iterations)', marker='s', markersize=3, color='darkred')
plt.plot(layersProposed_set, proposed_NMSE, ls='-', label='Proposed (48 Layers)', marker='o', markersize=3, color='darkblue')

plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (dB)')
plt.title('Error of recovered power spectrum varying SNRs')
plt.legend(loc='lower left', fontsize=6)
plt.savefig(f'../Test/Figures/testStatic/varying_dB.pdf')
plt.show()

plt.figure(dpi=800)
ISTA_2_DOA = [10*np.log10(i) for i in ISTA_2_DOA]
pro_DOA = [10*np.log10(i) for i in pro_DOA]
ISTA_1_DOA = [10*np.log10(i) for i in ISTA_1_DOA]
ISTA_14_DOA = [10*np.log10(i) for i in ISTA_14_DOA]
plt.plot(layersProposed_set, ISTA_2_DOA, ls='-', label='ISTA ($\\alpha=2$; 32 Iterations)', marker='x', markersize=3, color='darkred')
# plt.plot(layersProposed_set, ISTA_32, ls='-', label='ISTA ($\\alpha=1.5$)', marker='v', markersize=3, color='darkred')
plt.plot(layersProposed_set, ISTA_1_DOA, ls='-', label='ISTA ($\\alpha=1$; 96 Iterations)', marker='o', markersize=3, color='darkred')
# plt.plot(layersProposed_set, ISTA_12, ls='-', label='ISTA ($\\alpha=0.5$)', marker='^', markersize=3, color='darkred')
plt.plot(layersProposed_set, ISTA_14_DOA, ls='-', label='ISTA ($\\alpha=0.25$; 320 Iterations)', marker='s', markersize=3, color='darkred')
plt.plot(layersProposed_set, pro_DOA, ls='-', label='Proposed (48 Layers)', marker='o', markersize=3, color='darkblue')

plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (dB)')
plt.title('Error of estimated DOA varying SNRs')
plt.legend(loc='lower left', fontsize=6)
plt.savefig(f'../Test/Figures/testStatic/varying_dB_DOA.pdf')
plt.show()

