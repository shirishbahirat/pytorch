import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


size = 40

plt.rc('font', size=size)
plt.rc('axes', titlesize=size)
plt.rc('axes', labelsize=size)
plt.rc('xtick', labelsize=size)
plt.rc('ytick', labelsize=size)
plt.rc('legend', fontsize=size)
plt.rc('figure', titlesize=size)

'''
Figure 11 (Per-domain Bandwidth vs. Credit Refresh Rate)
x = [0.125, 0.25, 0.5, 0.75, 1]
HP Domain
y = [1399314,1307639,1127945,982021,705933]
MP Domain
y = [920473,901974,843761,769908,568635]
LP Domain
y = [464291,487710,519649,501006,393771]
VLP Domain
y = [159203,176234,211132,223110,210375]
'''

# HP Domain
y1 = [1399314, 1307639, 1127945, 982021, 705933]
# MP Domain
y2 = [920473, 901974, 843761, 769908, 568635]
# LP Domain
y3 = [464291, 487710, 519649, 501006, 393771]
# VLP Domain
y4 = [159203, 176234, 211132, 223110, 210375]


x = [0.125, 0.25, 0.5, 0.75, 1]

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(x, y1, label='HP', color='#FF00C4', marker='x', linestyle='dashed', linewidth=6, markersize=12)
plt.plot(x, y2, label='MP', color='#0D61F0', marker='o', linestyle='solid', linewidth=6, markersize=12)
plt.plot(x, y3, label='LP', color='#1B7F14', marker='d', linestyle='dotted', linewidth=6, markersize=12)
plt.plot(x, y4, label='VLP', color='#19CE25', marker='s', linestyle='dotted', linewidth=6, markersize=12)
plt.xlabel('Credit Refresh Rate')
plt.ylabel('Per-domain IO Count')
plt.ylim([00000, 2000000])
leg = plt.legend()
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")

plt.savefig('line.png', bbox_inches='tight', dpi=600)


plt.show()
