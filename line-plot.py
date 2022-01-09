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


x = np.linspace(0, 1, 20)

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)  # change width
    # ax.spines[axis].set_color('red')    # change color


plt.subplots_adjust(bottom=0.15)
plt.plot(x, x * 2, label='High', color='#FF00C4', marker='s', linestyle='dashed', linewidth=2, markersize=12)
plt.plot(x, x * 1.3, label='Medium', color='#0D61F0', marker='s', linestyle='solid', linewidth=2, markersize=12)
plt.plot(x, x * 1.5, label='Low', color='#1B7F14', marker='s', linestyle='dotted', linewidth=2, markersize=12)
plt.plot(x, x * 1.8, label='Very Low', color='#19CE25', marker='s', linestyle='dotted', linewidth=2, markersize=12)
plt.xlabel('Credit Refresh Rate')
plt.ylabel('Achieved BW')
leg = plt.legend()
leg.get_frame().set_linewidth(2.5)
leg.get_frame().set_edgecolor("black")

plt.savefig('line.png', bbox_inches='tight', dpi=600)


plt.show()
