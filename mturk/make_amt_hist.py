import csv, sys
from collections import defaultdict
import numpy as np

str2score = {'guaranteed': 4, 'probably': 3, 'unrelated': 2, 'unlikely': 1, 'impossible': 0, 'pjj-nonsense': 5, 'pw-nonsense': 6, 'wjj-nonsense': 7, 'word-nonsense': 8, 'nonsense': 9}
with open(sys.argv[1]) as f:
    r = csv.reader(f)
    next(r)
    hists = defaultdict(lambda: np.zeros(10))
    for row in r:
        for ann in row[3:]:
            hists[tuple(row[:3])][str2score[ann]] += 1

with open(sys.argv[2], 'w') as of:
    w = csv.writer(of)
    w.writerow(['whole', 'part', 'adj', 'impossible', 'unlikely', 'unrelated', 'probably', 'guaranteed', 'pjj-nonsense', 'part-whole-nonsense', 'whole-adj-nonsense', 'word-nonsense', 'nonsense'])
    for triple, hist in hists.items():
        print(triple, hist)
        to_write = list(triple) + hist.astype(np.int64).tolist()
        w.writerow(to_write)
