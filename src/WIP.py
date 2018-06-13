"""
    put random stuff I wrote in ipython that may be useful here. mostly PWKB stuff
"""
import copy
from collections import Counter
import operator

#lookups
whole2ss = {whole:wn.synsets(whole)[0] for whole in noun2parts.keys()}
ss2wholes = defaultdict(set)
for w,ss in whole2ss.items():
    ss2wholes[ss].add(w)

#keep only the "distinguishing" parts of a whole noun (so that 'actor' doesn't have all the body parts)
noun2parts_disting = copy.deepcopy(noun2parts)
for whole in wholes:
     paths = whole2ss[whole].hypernym_paths()
     for p in paths:
         for i,hyper in enumerate(p):
             #skip self-reference
             if i == len(p) - 1:
                 continue
             if hyper in whole_ss:
                 for hyper_whole in ss2wholes[hyper]:
                     noun2parts_disting[whole] = noun2parts_disting[whole] - noun2parts_disting[hyper_whole]

#doc frequencies (to catch things like 'chromosome' that persist)
dfs = Counter()
for noun, parts in noun2parts_disting.items():
     for part in parts:
         dfs[part] += 1
sorted(dfs.items(), key=operator.itemgetter(1), reverse=True)[:30]
