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


x = [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]
# HP Domain
# Simulation
y11 = [182, 283, 404, 513, 620, 705]
# QoS Calculator
y12 = [170, 255, 425, 510, 595, 680]

# MP Domain
# Simulation
y21 = [199, 829, 1979, 3169, 3370, 3707]
# QoS Calculator
y22 = [170, 850, 1955, 3145, 3325, 3675]

# LP Domain
# Simulation
y31 = [186, 847, 2069, 3302, 4303, 4937]
# QoS Calculator
y32 = [170, 850, 2040, 3315, 4335, 4930]

# VLP Domain
# Simulation
y41 = [178, 839, 2101, 3677, 4827, 6252]
# QoS Calculator
y42 = [170, 850, 2125, 3655, 4845, 6290]


fig = plt.figure(figsize=(9, 9))

ax = fig.add_subplot(221)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(x, y11, label='Simulation', color='#DA0A0A', marker='s', linestyle='solid', linewidth=6, markersize=12)
plt.plot(x, y12, label='QoS Calculator', color='#6C4BEE', marker='d', linestyle='dashed', linewidth=6, markersize=12)
#plt.xlabel('Tail Latency Percentiles')
plt.ylabel('Latency (us)')
leg = plt.legend()
plt.ylim([0, 6500])
plt.text(.6, 2000, 'HP Domain', fontsize=25)
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")


ax = fig.add_subplot(222)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(x, y21, label='Simulation', color='#0A18DA', marker='s', linestyle='solid', linewidth=6, markersize=12)
plt.plot(x, y22, label='QoS Calculator', color='#5DDA0A', marker='d', linestyle='dashed', linewidth=6, markersize=12)
#plt.xlabel('Tail Latency Percentiles')
#plt.ylabel('Latency (us)')
leg = plt.legend()
plt.ylim([0, 6500])
plt.text(.6, 2000, 'MP Domain', fontsize=25)
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")


plt.savefig('line.png', bbox_inches='tight', dpi=600)


ax = fig.add_subplot(223)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(x, y31, label='Simulation', color='#0A94DA', marker='s', linestyle='solid', linewidth=6, markersize=12)
plt.plot(x, y32, label='QoS Calculator', color='#CCDA0A', marker='d', linestyle='dashed', linewidth=6, markersize=12)
plt.xlabel('Tail Latency Percentiles')
plt.ylabel('Latency (us)')
plt.ylim([0, 6500])
leg = plt.legend()
plt.text(.6, 2000, 'LP Domain', fontsize=25)
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")


ax = fig.add_subplot(224)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(x, y41, label='Simulation', color='#DAA90A', marker='s', linestyle='solid', linewidth=6, markersize=12)
plt.plot(x, y42, label='QoS Calculator', color='#DA250A', marker='d', linestyle='dashed', linewidth=6, markersize=12)
plt.xlabel('Tail Latency Percentiles')
#plt.ylabel('Latency (us)')
leg = plt.legend()
plt.ylim([0, 6500])
plt.text(.6, 2000, 'VLP Domain', fontsize=25)
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")


plt.show()

#
