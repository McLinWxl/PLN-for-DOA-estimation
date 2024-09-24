#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import matplotlib.pyplot as plt
import numpy as np

layersProposed_set = [2, 4, 8, 16, 24, 32, 48]

proposed_NMSE = [-1.8798938393592834, -2.619062066078186, -5.331571102142334, -12.340117692947388, -20.416455268859863, -23.547139167785645, -23.698501586914062]
ISTA_1 = [-0.7125440239906311, -0.9332600980997086, -1.3760267198085785, -2.377934604883194, -3.4298446774482727, -5.327702164649963, -11.596786975860596]
ISTA_32 = [-0.9091171622276306, -1.2669084966182709, -2.012305110692978, -3.725842237472534, -7.552184462547302, -11.215542554855347, -12.102879285812378]
ISTA_2 = [-1.2430433928966522, -1.7579562962055206, -2.870059609413147, -6.7433130741119385, -10.575196743011475, -10.767818689346313, -11.030197143554688]
ISTA_12 = [-0.574607364833355, -0.6873902678489685, -0.8835042268037796, -1.2743683159351349, -1.6941513121128082, -2.1827273070812225, -3.191394805908203]
ISTA_14 = [-0.5196017399430275, -0.5915206298232079, -0.6984639912843704, -0.8747684210538864, -1.219390630722046, -1.219390630722046, -1.6017396748065948]

plt.style.use(['science', 'ieee', 'grid'])
plt.figure(dpi=800)

plt.plot(layersProposed_set, ISTA_2, ls='-', label='ISTA ($\\alpha=2$)', marker='x', markersize=3, color='darkred')
plt.plot(layersProposed_set, ISTA_32, ls='-', label='ISTA ($\\alpha=1.5$)', marker='v', markersize=3, color='darkred')
plt.plot(layersProposed_set, ISTA_1, ls='-', label='ISTA ($\\alpha=1$)', marker='o', markersize=3, color='darkred')
plt.plot(layersProposed_set, ISTA_12, ls='-', label='ISTA ($\\alpha=0.5$)', marker='^', markersize=3, color='darkred')
plt.plot(layersProposed_set, ISTA_14, ls='-', label='ISTA ($\\alpha=0.25$)', marker='s', markersize=3, color='darkred')
plt.plot(layersProposed_set, proposed_NMSE, ls='-', label='Proposed', marker='o', markersize=3, color='darkblue')

plt.xlabel('Iterations / Layers ($k$)')
plt.ylabel('NMSE (dB)')
plt.title('Convergence of Proposed and ISTAs')
plt.legend(loc='lower left', fontsize=6)
plt.savefig(f'../Test/Figures/testStatic/inf_dB.pdf')
plt.show()

plt.figure(dpi=800)

plt.plot(layersProposed_set, ISTA_2, ls='-', label='ISTA ($\\alpha=2$)', marker='x', markersize=3, color='darkred')
plt.plot(layersProposed_set + [64], ISTA_32 + [-12.244411706924438], ls='-', label='ISTA ($\\alpha=1.5$)', marker='v', markersize=3, color='darkred')
plt.plot(layersProposed_set + [64, 96, 128], ISTA_1 +[-13.774442672729492, -14.082465171813965, -13.957077264785767], ls='-', label='ISTA ($\\alpha=1$)', marker='o', markersize=3, color='darkred')
plt.plot(layersProposed_set + [64, 96, 128, 192, 256], ISTA_12 + [-4.809447228908539, -12.735238075256348, -18.782862424850464, -20.83617925643921, -20.097599029541016], ls='-', label='ISTA ($\\alpha=0.5$)', marker='^', markersize=3, color='darkred')
plt.plot(layersProposed_set + [64, 96, 128, 192, 256, 320, 384], ISTA_14 + [-2.022893726825714, -3.00348162651062, -4.351447820663452, -13.148527145385742, -21.46085262298584, -24.34135913848877, -24.52937602996826], ls='-', label='ISTA ($\\alpha=0.25$)', marker='s', markersize=3, color='darkred')
plt.plot(layersProposed_set, proposed_NMSE, ls='-', label='Proposed', marker='o', markersize=3, color='darkblue')
plt.xlabel('Iterations / Layers ($k$)')
plt.ylabel('NMSE (dB)')
plt.title('Convergence of Proposed and ISTAs')
plt.legend(fontsize=6, loc='upper right')
plt.savefig(f'../Test/Figures/testStatic/inf_dB_long.pdf')
plt.show()

