'''

x = [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]

# HP QoS
#"Refresh Rate = 0.125"
y = [193, 335, 507, 763, 1053, 1399]
#"Refresh Rate = 0.25"
y = [183, 329, 500, 753, 1024, 1252]
#"Refresh Rate = 0.5"
y = [165, 314, 482, 727, 997, 1185]
#"Refresh Rate = 0.75"
y = [149, 294, 449, 671, 908, 1207]
#"Refresh Rate = 1"
y = [120, 255, 398, 551, 697, 923]

# VLP QoS
#"Refresh Rate = 0.125"
y = [179, 1243, 4184, 7948, 12061, 13801]
#"Refresh Rate = 0.25"
y = [171, 1023, 3369, 6135, 8657, 9914]
#"Refresh Rate = 0.5"
y = [155, 670, 2136, 3889, 6843, 9386]
#"Refresh Rate = 0.75"
y = [140, 462, 1357, 2415, 3760, 5166]
#"Refresh Rate = 1"
y = [111, 311, 655, 1084, 1562, 2354

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

# HP QoS
x = [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]
#"Refresh Rate = 0.125"
y11 = [193, 335, 507, 763, 1053, 1399]
#"Refresh Rate = 0.25"
y12 = [183, 329, 500, 753, 1024, 1252]
#"Refresh Rate = 0.5"
y13 = [165, 314, 482, 727, 997, 1185]
#"Refresh Rate = 0.75"
y14 = [149, 294, 449, 671, 908, 1207]
#"Refresh Rate = 1"
y15 = [120, 255, 398, 551, 697, 923]

# VLP QoS
#"Refresh Rate = 0.125"
y21 = [179, 1243, 4184, 7948, 12061, 13801]
#"Refresh Rate = 0.25"
y22 = [171, 1023, 3369, 6135, 8657, 9914]
#"Refresh Rate = 0.5"
y23 = [155, 670, 2136, 3889, 6843, 9386]
#"Refresh Rate = 0.75"
y24 = [140, 462, 1357, 2415, 3760, 5166]
#"Refresh Rate = 1"
y25 = [111, 311, 655, 1084, 1562, 2354


fig = plt.figure(figsize=(9, 9))

ax = fig.add_subplot(211)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(x, y11, label='Simulation', color='#DA0A0A', marker='s', linestyle='solid', linewidth=6, markersize=12)
plt.plot(x, y12, label='QoS Calculator', color='#6C4BEE', marker='d', linestyle='dashed', linewidth=6, markersize=12)
plt.plot(x, y13, label='Simulation', color='#DA0A0A', marker='s', linestyle='solid', linewidth=6, markersize=12)
plt.plot(x, y14, label='QoS Calculator', color='#6C4BEE', marker='d', linestyle='dashed', linewidth=6, markersize=12)
plt.plot(x, y15, label='Simulation', color='#DA0A0A', marker='s', linestyle='solid', linewidth=6, markersize=12)

# plt.xlabel('Tail Latency Percentiles')
plt.ylabel('Latency (us)')
leg = plt.legend()
plt.ylim([0, 15000])
plt.text(.6, 2000, 'HP Domain', fontsize=25)
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")


ax = fig.add_subplot(212)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(x, y21, label='Simulation', color='#DA0A0A', marker='s', linestyle='solid', linewidth=6, markersize=12)
plt.plot(x, y22, label='QoS Calculator', color='#6C4BEE', marker='d', linestyle='dashed', linewidth=6, markersize=12)
plt.plot(x, y23, label='Simulation', color='#DA0A0A', marker='s', linestyle='solid', linewidth=6, markersize=12)
plt.plot(x, y24, label='QoS Calculator', color='#6C4BEE', marker='d', linestyle='dashed', linewidth=6, markersize=12)
plt.plot(x, y25, label='Simulation', color='#DA0A0A', marker='s', linestyle='solid', linewidth=6, markersize=12)
# plt.xlabel('Tail Latency Percentiles')
# plt.ylabel('Latency (us)')
leg = plt.legend()
plt.ylim([0, 15000])
plt.text(.6, 2000, 'MP Domain', fontsize=25)
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")


plt.show()
