#!/usr/bin/env python3

# An ad-hoc script for plotting tables copy-pasted from LibreOffice Calc

import fileinput
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.transforms import Affine2D

group_names = []
values = []

plt.rcParams['font.family'] = 'serif'
plt.rcParams['savefig.format'] = 'pdf'
plt.figure(figsize=(5,4))

for line in fileinput.input():
	field = line.rstrip().split('\t')
	group_names.append(field[0])
	values.append(field[1:])

feature_names = values[0]
group_names = group_names[1:]
values = values[1:]

print (group_names)
print (values)
values = np.array(values, dtype=float)
values[values < 0] = 0
values /= np.sum(values,axis=0)

width = 0.95 / values.shape[1]

order = np.argsort(-np.mean(values, axis=1))
x = np.arange(len(group_names))
for i in range(values.shape[1]):
	plt.bar(x +width*(i-values.shape[1]/2), values[order,i], width, label=feature_names[i])
plt.legend()
plt.xticks(x, [group_names[i] for i in order], rotation=45, horizontalalignment='right')
for tick in plt.gca().xaxis.get_major_ticks():
	label = tick.label1
	label.set_transform(label.get_transform() + Affine2D().translate(8,0))
plt.ylabel('Normalized metric')
plt.tight_layout()

plt.savefig(sys.stdout.buffer, bbox_inches = 'tight', pad_inches = 0)
plt.show()

