'''

Figure 10 (Multi Domain)
x = [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]

HP Domain
#Simulation
y = [182, 283, 404, 513, 620, 705]
#QoS Calculator
y = [170, 255, 425, 510, 595, 680]

MP Domain
#Simulation
y = [199, 829, 1979, 3169, 3870, 4707]
#QoS Calculator
y = [170, 850, 1955, 3145, 3825, 4675]

LP Domain
#Simulation
y = [186, 847, 2069, 3302, 4303, 4937]
#QoS Calculator
y = [170, 850, 2040, 3315, 4335, 4930]

VLP Domain
#Simulation
y = [178, 839, 2101, 3677, 4827, 6252]
#QoS Calculator
y = [170, 850, 2125, 3655, 4845, 6290]
'''


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


size = 25

plt.rc('font', size=size)
plt.rc('axes', titlesize=size)
plt.rc('axes', labelsize=size)
plt.rc('xtick', labelsize=size)
plt.rc('ytick', labelsize=size)
plt.rc('legend', fontsize=size)
plt.rc('figure', titlesize=size)


x = np.linspace(.5, 1, 6)
y1 = [89, 120, 130, 150, 190, 520]
y2 = [85, 122, 135, 156, 193, 526]


# Simulation
y = [89, 171, 259, 350, 439, 521]
# QoS Calculator
y = [85, 170, 255, 340, 425, 510]


xnew = np.linspace(.5, 1, num=50, endpoint=True)
y1new = interp1d(x, y1, kind='cubic')
y2new = interp1d(x, y2, kind='cubic')

fig = plt.figure(figsize=(9, 9))

ax = fig.add_subplot(221)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(xnew, y1new(xnew), label='Simulation', color='#2519CE', marker='s', linestyle='dashed', linewidth=2, markersize=12)
plt.plot(xnew, y2new(xnew), label='QoS Calculator', color='#CE4919', marker='s', linestyle='solid', linewidth=2, markersize=12)
#plt.xlabel('Tail Latency Percentiles')
plt.ylabel('Latency (us)')
leg = plt.legend()
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")


ax = fig.add_subplot(222)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(xnew, y1new(xnew), label='Simulation', color='#2519CE', marker='s', linestyle='dashed', linewidth=2, markersize=12)
plt.plot(xnew, y2new(xnew), label='QoS Calculator', color='#CE4919', marker='s', linestyle='solid', linewidth=2, markersize=12)
#plt.xlabel('Tail Latency Percentiles')
#plt.ylabel('Latency (us)')
leg = plt.legend()
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")


plt.savefig('line.png', bbox_inches='tight', dpi=600)


ax = fig.add_subplot(223)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(xnew, y1new(xnew), label='Simulation', color='#2519CE', marker='s', linestyle='dashed', linewidth=2, markersize=12)
plt.plot(xnew, y2new(xnew), label='QoS Calculator', color='#CE4919', marker='s', linestyle='solid', linewidth=2, markersize=12)
plt.xlabel('Tail Latency Percentiles')
plt.ylabel('Latency (us)')

leg = plt.legend()
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")


ax = fig.add_subplot(224)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(xnew, y1new(xnew), label='Simulation', color='#2519CE', marker='s', linestyle='dashed', linewidth=2, markersize=12)
plt.plot(xnew, y2new(xnew), label='QoS Calculator', color='#CE4919', marker='s', linestyle='solid', linewidth=2, markersize=12)
plt.xlabel('Tail Latency Percentiles')
#plt.ylabel('Latency (us)')
leg = plt.legend()
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")


plt.show()
