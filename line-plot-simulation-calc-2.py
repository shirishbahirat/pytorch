import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


size = 40

plt.rc('font', size=size)
plt.rc('axes', titlesize=size)
plt.rc('axes', labelsize=size)
plt.rc('xtick', labelsize=size)
plt.rc('ytick', labelsize=size)
plt.rc('legend', fontsize=size)
plt.rc('figure', titlesize=size)


x = [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]
# Simulation
y1 = [89, 171, 259, 350, 439, 521]
# QoS Calculator
y2 = [85, 170, 255, 340, 425, 510]

#


# Simulation
y = [89, 171, 259, 350, 439, 521]
# QoS Calculator
y = [85, 170, 255, 340, 425, 510]


fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(111)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(x, y1, label='Simulation', color='#DA0A0A', marker='s', linestyle='solid', linewidth=6, markersize=12)
plt.plot(x, y2, label='QoS Calculator', color='#6C4BEE', marker='d', linestyle='dashed', linewidth=6, markersize=12)
plt.xlabel('Tail Latency Percentiles')
plt.ylabel('Latency (us)')
leg = plt.legend()
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")

plt.savefig('line.png', bbox_inches='tight', dpi=600)


plt.show()
