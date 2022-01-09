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


x = np.linspace(.5, 1, 6)
y1 = [89, 120, 135, 160, 190, 520]
y2 = [85, 122, 138, 156, 193, 526]


# Simulation
y = [89, 171, 259, 350, 439, 521]
# QoS Calculator
y = [85, 170, 255, 340, 425, 510]


xnew = np.linspace(.5, 1, num=50, endpoint=True)
y1new = interp1d(x, y1, kind='cubic')
y2new = interp1d(x, y2, kind='cubic')

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(xnew, y1new(xnew), label='Simulation', color='#A5BEE9', marker='s', linestyle='dashed', linewidth=2, markersize=12)
plt.plot(xnew, y2new(xnew), label='QoS Calculator', color='#6C4BEE', marker='d', linestyle='solid', linewidth=2, markersize=12)
plt.xlabel('Tail Latency Percentiles')
plt.ylabel('Latency (us)')
leg = plt.legend()
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")

plt.savefig('line.png', bbox_inches='tight', dpi=600)


plt.show()
