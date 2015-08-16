__author__ = 'Freeman'

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

with open('C:\Users\Freeman\Downloads\Test for web-analyst (2).csv', 'rb') as rh:
    reader = rh.readlines()
    for line in reader:
        l = line.split('\r')

l_data = []
for i in l[1:]:
    values = [int(_) for _ in i.split(';')]
    l_data.append(values[1:])

names = l[0].split(';')[1:]

##### preprocessing ######
freq_array = np.array(l_data)
low_val_array = freq_array < 5
freq_array[low_val_array] = 0
freq_array = freq_array[freq_array.sum(axis = 1) != 0]
zero_array_mask = freq_array != 0
freq_array_log = np.log(freq_array+1)


tbl = np.zeros((30,30), dtype=float)
for i in range(freq_array_log.shape[1]):
    for n in range(freq_array_log.shape[1]):
        # mask = (one_table[:,i] != 0) | (one_table[:,n] != 0)
        # mask = np.array(mask)[:,0]
        a = freq_array_log[:,i]
        b = freq_array_log[:,n]
        TP, TF, FP, FN = ((a > 0) & (b > 0)).sum(), ((a == 0) & (b == 0)).sum(), ((a > 0) & (b == 0)).sum(), ((a == 0) & (b > 1)).sum()
        r, cr = pearsonr(a, b)
        print (r, round(cr,2), TP, TF, FP, FN, i, n)
        if cr < 0.05:
            if r > abs(0.25):
                if i != n:
                    tbl[i, n] = round(r, 2)


ax1 = plt.subplot()
ax1.set_yticks(np.arange(tbl.shape[0])+0.5, minor=False)
ax1.set_xticks(np.arange(tbl.shape[1])+0.5, minor=False)

ax1.set_xticklabels(names, minor=False)
ax1.set_yticklabels(names, minor=False)
plot1 = ax1.pcolormesh(tbl)
cbar1 = plt.colorbar(plot1)