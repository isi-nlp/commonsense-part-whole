"""
    Turn a tuple into a context hypothesis pair with simple programmatic language
"""
import csv, sys

with open(sys.argv[1]) as f:
    with open(sys.argv[2], 'w') as of:
        r = csv.reader(f)
        w = csv.writer(of)
        w.writerow(['context', 'hypothesis', 'label', 'bin_label'])
        #header
        next(r)
        for row in r:
            whole, part, jj = tuple(row[:3])
            det = 'an' if jj[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
            context = 'There is %s %s %s.' % (det, jj, whole)
            #context = 'There is a %s %s which has a %s.' % (jj, whole, part)
            hypothesis = "The %s's %s is %s." % (whole, part, jj)
            #hypothesis = "The %s has a %s which is %s." % (whole, part, jj)
            w.writerow([context, hypothesis, row[3], row[4]])
