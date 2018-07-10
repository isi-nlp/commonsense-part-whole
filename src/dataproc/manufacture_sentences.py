"""
    Turn a tuple into a context hypothesis pair with simple programmatic language
"""
import argparse, csv, sys

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="_train file with triples")
parser.add_argument("outfile", type=str, help="base name of output files")
args = parser.parse_args()

for fold in ['train', 'dev', 'test']:
    with open(args.file.replace('train', fold)) as f:
        with open(args.outfile.replace('train', fold), 'w') as of:
            r = csv.reader(f)
            w = csv.writer(of, delimiter='\t')
            w.writerow(['whole', 'part', 'jj', 'hypothesis', 'context', 'label', 'bin_label'])
            #header
            next(r)
            for row in r:
                whole, part, jj = tuple(row[:3])
                det = 'an' if jj[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
                context = 'There is %s %s %s.' % (det, jj, whole)
                #context = 'There is a %s %s which has a %s.' % (jj, whole, part)
                hypothesis = "The %s's %s is %s." % (whole, part, jj)
                #hypothesis = "The %s has a %s which is %s." % (whole, part, jj)
                w.writerow([whole, part, jj, hypothesis, context, row[3], row[4]])
