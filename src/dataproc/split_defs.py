"""
    Split the definition data
"""
import argparse, csv, json, random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to wordnet definition data')
    parser.add_argument('outfile', type=str, help='path to write split data. base name: will append _train, _dev, _test')
    args = parser.parse_args()

    folds = ['train', 'dev', 'test']
    trip2split = {} #(whole, part, adj): split
    for fold in folds:
        with open('../../data/annotated/full_%s.csv' % fold) as f:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                tup = tuple(row[:3])
                trip2split[tup] = fold

    trf = open(args.outfile.replace('.csv', '_train.csv'), 'w')
    dvf = open(args.outfile.replace('.csv', '_dev.csv'), 'w')
    tef = open(args.outfile.replace('.csv', '_test.csv'), 'w')

    trw = csv.writer(trf, delimiter='\t')
    trw.writerow(['whole', 'part', 'jj', 'label', 'bin_label', 'whole_def', 'part_def', 'jj_def'])
    dvw = csv.writer(dvf, delimiter='\t')
    dvw.writerow(['whole', 'part', 'jj', 'label', 'bin_label', 'whole_def', 'part_def', 'jj_def'])
    tew = csv.writer(tef, delimiter='\t')
    tew.writerow(['whole', 'part', 'jj', 'label', 'bin_label', 'whole_def', 'part_def', 'jj_def'])
    
    with open(args.file) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            trip = tuple(row[:3])
            if trip2split[trip] == 'train':
                trw.writerow(row)
            elif trip2split[trip] == 'dev':
                dvw.writerow(row)
            elif trip2split[trip] == 'test':
                tew.writerow(row)

    trf.close()
    dvf.close()
    tef.close()
