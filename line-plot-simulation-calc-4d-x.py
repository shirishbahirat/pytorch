import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


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
