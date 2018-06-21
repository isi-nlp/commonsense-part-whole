"""
    Post-process triples to limit colors to one per part-whole
"""
from collections import defaultdict
import csv

COLORS = set(['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'violet', 'black', 'white', 'brown', 'gray', 'grey', 'pink'])
with open('/home/jamesm/commonsense-part-whole/data/candidates/vis_uniq.csv') as f:
    with open('/home/jamesm/commonsense-part-whole/data/candidates/vis_uniq_less_colors.csv', 'w') as of:
        r = csv.reader(f)
        w = csv.writer(of)
        pw2jjs = defaultdict(set)
        for row in r:
            whole, part, jj = row[0], row[1], row[2]
            pw2jjs[(whole, part)].add(jj)

        for (whole, part), jjs in pw2jjs.items():
            found_color = False
            for jj in jjs:
                if jj in COLORS:
                    if not found_color:
                        w.writerow([whole, part, jj])
                        found_color = True
                else:
                    w.writerow([whole, part, jj])
