import numpy as np
import matplotlib.pyplot as plt

barWidth = 0.25

size = 20

plt.rc('font', size=size)
plt.rc('axes', titlesize=size)
plt.rc('axes', labelsize=size)
plt.rc('xtick', labelsize=size)
plt.rc('ytick', labelsize=size)
plt.rc('legend', fontsize=size)
plt.rc('figure', titlesize=size)


domain1_qos = [[293, 355, 412, 578, 742, 922, 1385, 4752],
               [143, 194, 253, 416, 570, 725, 881, 1037],
               [113, 184, 202, 351, 502, 644, 791, 947]]

r1 = np.arange(len(domain1_qos[0]))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]


def plot_data(name, dqos, xlbl, ylbl):
    plt.bar(r1, dqos[2], color='#5d7f5e', width=barWidth, edgecolor='white', label='End to end priority')
    plt.bar(r2, dqos[1], color='#85af2d', width=barWidth, edgecolor='white', label='Controlled forwarding')
    plt.bar(r3, dqos[0], color='#a6d512', width=barWidth, edgecolor='white', label='No priority')

    plt.xlabel(xlbl, fontweight='bold')
    plt.ylabel(ylbl, fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(dqos[0]))], ['0.5', '0.75', '1 9', '2 9', '3 9', '4 9', '5 9', '6 9'])

    plt.title(name, y=1.0, pad=-130)

    plt.legend()


fig = plt.figure()
ax = fig.add_subplot(221)

plot_data('(a) High Priority', domain1_qos, '', 'QoS [us]')


domain2_qos = [[293, 355, 412, 578, 742, 930, 1385, 3130],
               [306, 351, 408, 570, 725, 881, 1045, 1045],
               [128, 196, 273, 482, 701, 914, 1139, 1352]]

ax = fig.add_subplot(222)
plot_data('(b) Mid Priority', domain2_qos, '', '')

domain3_qos = [[293, 355, 412, 578, 742, 930, 1385, 3130],
               [627, 668, 717, 873, 1037, 1205, 1385, 1582],
               [165, 237, 347, 644, 955, 1287, 1647, 2278]]

ax = fig.add_subplot(223)
plot_data('(c) Low Priority', domain3_qos, 'Latency Percentiles', 'QoS [us]')


domain4_qos = [[293, 355, 408, 578, 742, 930, 1385, 5014],
               [1270, 1303, 1352, 1516, 1680, 1860, 2147, 2376],
               [192, 281, 416, 783, 1172, 1582, 2073, 2606]]

ax = fig.add_subplot(224)
plot_data('(d) Very Low Priority', domain4_qos, 'Latency Percentiles', '')

plt.savefig('domain2.png', bbox_inches='tight', dpi=1600)
plt.show()
