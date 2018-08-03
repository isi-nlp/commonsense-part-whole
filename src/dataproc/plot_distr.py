"""
    Plot chart of label distribution
    Hopefully this bar is more interpretable than a pie graph
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

counts = [0] * 5
with open('../../data/annotated/distr_two_agree.txt') as f:
    for line in f:
        line = line.strip().split()
        counts[int(line[1])] = int(line[0])
    
fig = plt.figure()

colors = ['cornflowerblue', 'cornflowerblue', 'tomato', 'tomato', 'tomato']
labels = ['Guaranteed', 'Probably', 'Unrelated', 'Unlikely', 'Impossible']
saturations = [1, 0.5, 0.33, 0.67, 1]
vals = np.cumsum(counts)
locs = vals - np.array(counts) / 2 - 100

for ix,i in enumerate(range(-1,-6,-1)):
    g = sns.barplot([vals[i]], color=colors[ix], saturation=saturations[ix], label=labels[ix])
    g.text(locs[i], 0, "{}\n{:2.1f}%".format(counts[i],counts[i]/sum(counts)*100), color='white')
    
fig.set_size_inches(9,4);
plt.xlabel('Examples (total={})'.format(sum(counts)));
plt.title('Label distribution');
plt.legend(loc='lower right');
plt.tight_layout();
plt.show()
