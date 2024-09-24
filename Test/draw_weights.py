#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

with open('../Test/L16-V0/Weights/weights_best.txt', 'r') as file:
    data = file.readlines()
    theta = data[2]
    gamma = data[4]
    # theta = '[0.1, 0.2, 0.3]' of str, convert to list
    theta = theta[1:-3].split(', ')
    theta = [(float(i)) for i in theta]
    gamma = gamma[2:-1].split(', ')
    gamma = [(float(i)) for i in gamma]

    lambda_ = [theta[i]/gamma[i] for i in range(len(theta))]

import matplotlib.pyplot as plt
plt.style.use(['science', 'ieee', 'grid'])

# x_label =
# loss = [0.06695171445608139, 0.047059737145900726, 0.029464442282915115, 0.011734290048480034, 0.009682554751634598, 0.010060031898319721, 0.009174397215247154, 0.00947522185742855, 0.00983314961194992, 0.00983314961194992, 0.009563836269080639]
fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)

lns1 = ax.plot(theta, '-', label = 'Thresholds')
ax2 = ax.twinx()
lns3 = ax2.plot(gamma, '-r', label = 'Step size')

# added these three lines
lns = lns1+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.grid()
ax.set_xlabel("Layers")
ax.set_ylabel("Thresholds")
ax2.set_ylabel("Step size")
# ax2.set_ylim(-0.5, 0.5)
# ax.set_ylim(-0.5, 0.5)
# plt.savefig('0.png')
# plt.plot(theta, label='Thresholds')
# plt.plot(gamma, label='Step size')
# plt.savefig('../Test/Figures/weights.pdf')
# plt.legend()
plt.show()

plt.figure(dpi=800)
plt.plot(lambda_[:], label='Lambda')
plt.legend()
plt.show()


