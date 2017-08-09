#!/usr/bin/env python

import sys
import numpy as np

#rc('text',usetex=1)
import numpy as np
import matplotlib.pyplot as plt

outfile='plot.pdf'


N = 4
ncvo_res = (77.6, 71.5, 58, 58.4)
our_res =   (91.8, 77.05, 71.78, 94.0)

ind = np.array((0.025, 0.275, 0.525, 0.775))  # the x locations for the groups
width = 0.1       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, ncvo_res, width, color='#FCC200')
rects2 = ax.bar(ind+width, our_res, width, color='#800000')

# add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy (%)', fontsize=20)
#ax.set_title('Classification accuracy')
ax.set_xticks(ind+width)
#ax.xticks(ind+width, ('Type', 'Source', 'Joint Type + Source', 'Type'), font=12, rotation=45)
ax.set_xticklabels(('Type', 'Source', 'Type + Source', 'Type'), ind+width, fontsize=14)


ax.set_yticklabels(ax.get_yticks(), fontsize=14) 

# Shrink current axis's height by 10% on the bottom so that legend can fit
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend((rects1[0], rects2[0]), ('Baseline NCVO accuracy', 'S2DS accuracy'), loc='upper center', bbox_to_anchor=(0.5, -0.05),
          ncol=2, fancybox=True, shadow=True) #

#ax.legend( , ('Baseline NCVO accuracy', 'S2DS accuracy') )
plt.axvline(x=0.75, color='k')

ax.text(0.29, 101, "I N C O M E", fontsize='16')
ax.text(0.75, 101, "EXPENDITURE", fontsize='16')
def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.0*height, '%d'%int(height),
                ha='center', va='bottom', fontsize=16)

autolabel(rects1)
autolabel(rects2)



plt.show()