import matplotlib.pyplot as plt
VERY_SMALL_SIZE = 6
SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

line_width = 2

figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# domain1
ax1.plot([0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], [293, 355, 412, 578, 742, 922, 1385, 4752], 'r', linewidth=line_width, marker='x', linestyle='dotted')
ax1.plot([0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], [143, 194, 253, 416, 570, 725, 881, 1037], 'b', linewidth=line_width, marker='d', linestyle='solid')
ax1.plot([0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], [113, 184, 202, 351, 502, 644, 791, 947], 'g', linewidth=line_width, marker='o', linestyle='-.')
# domain2
ax2.plot([0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], [293, 355, 412, 578, 742, 930, 1385, 3130], 'r', linewidth=line_width, marker='x', linestyle='dotted')
ax2.plot([0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], [306, 351, 408, 570, 725, 881, 1045, 1045], 'b', linewidth=line_width, marker='d', linestyle='solid')
ax2.plot([0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], [128, 196, 273, 482, 701, 914, 1139, 1352], 'g', linewidth=line_width, marker='o', linestyle='-.')
# domain3
ax3.plot([0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], [293, 355, 412, 578, 742, 930, 1385, 3130], 'r', linewidth=line_width, marker='x', linestyle='dotted')
ax3.plot([0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], [627, 668, 717, 873, 1037, 1205, 1385, 1582], 'b', linewidth=line_width, marker='d', linestyle='solid')
ax3.plot([0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], [165, 237, 347, 644, 955, 1287, 1647, 2278], 'g', linewidth=line_width, marker='o', linestyle='-.')
# domain4
ax4.plot([0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], [293, 355, 408, 578, 742, 930, 1385, 5014], 'r', linewidth=line_width, marker='x', linestyle='dotted')
ax4.plot([0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], [1270, 1303, 1352, 1516, 1680, 1860, 2147, 2376], 'b', linewidth=line_width, marker='d', linestyle='solid')
ax4.plot([0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], [192, 281, 416, 783, 1172, 1582, 2073, 2606], 'g', linewidth=line_width, marker='o', linestyle='-.')

ax1.set_title("(a) High Priority", y=1.0, pad=-20)
#ax1.legend(["No Priority", "Controlled Forwarding", "End-to-End Priority"], loc ="upper left", pad=-14)
ax1.set_ylabel('Latency (us)')
ax1.set_xlabel('Latency Percentiles')


ax2.set_title("(b) Mid Priority", y=1.0, pad=-20)
#ax2.legend(["No Priority", "Controlled Forwarding", "End-to-End Priority"], loc ="upper left")
ax2.set_ylabel('Latency (us)')
ax2.set_xlabel('Latency Percentiles')

ax3.set_title("(c) Low Priority", y=1.0, pad=-20)
#ax3.legend(["No Priority", "Controlled Forwarding", "End-to-End Priority"], loc ="upper left")
ax3.set_ylabel('Latency (us)')
ax3.set_xlabel('Latency Percentiles')

ax4.set_title("(d) Very Low Priority", y=1.0, pad=-20)
#ax4.legend(["No Priority", "Controlled Forwarding", "End-to-End Priority"], loc ="upper left")
ax4.set_ylabel('Latency (us)')
ax4.set_xlabel('Latency Percentiles')

figure.tight_layout(pad=0.8)
figure.legend(["No Priority", "Controlled Forwarding", "End-to-End Priority"], loc="lower center", ncol=3, borderpad=-.5)
#plt.figtext(.15, .15, "QD = [32(HP),24(MP),16(LP),8(VLP)], Dies = 32")
ax1.set_ylim([0, 1200])
ax2.set_ylim([0, 1200])
ax3.set_ylim([0, 2500])
ax4.set_ylim([0, 2650])
plt.savefig('real_impl_qos.jpg', bbox_inches='tight', dpi=300)
plt.show()
