"""
    Read snli style sentence data for each part-whole-adjective. 
    Copy labels and splits from that that exists for triple datasets
"""
import argparse, csv, random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to snli style sentences data')
    parser.add_argument('outfile', type=str, help='path to write annotated data. base name: will append _train, _dev, _test')
    args = parser.parse_args()

    folds = ['train', 'dev', 'test']
    trip2labels = {} #(whole, part, adj): (split, label, bin_label)
    for fold in folds:
        with open('../../data/annotated/full_%s.csv' % fold) as f:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                tup = tuple(row[:3])
                trip2labels[tup] = (fold, row[3], row[4])

    trf = open(args.outfile.replace('.csv', '_train.csv'), 'w')
    dvf = open(args.outfile.replace('.csv', '_dev.csv'), 'w')
    tef = open(args.outfile.replace('.csv', '_test.csv'), 'w')

    trw = csv.writer(trf, delimiter='\t')
    trw.writerow(['whole', 'part', 'jj', 'hypothesis', 'context', 'label', 'bin_label'])
    dvw = csv.writer(dvf, delimiter='\t')
    dvw.writerow(['whole', 'part', 'jj', 'hypothesis', 'context', 'label', 'bin_label'])
    tew = csv.writer(tef, delimiter='\t')
    tew.writerow(['whole', 'part', 'jj', 'hypothesis', 'context', 'label', 'bin_label'])
    
    with open(args.file) as f:
        r = csv.reader(f, delimiter='\t')
        #header
        next(r)
        for row in r:
            whole, part, jj = tuple(row[:3])
            if (whole, part, jj) in trip2labels:
                split, label, bin_label = trip2labels[(whole, part, jj)]
                if split == 'train':
                    trw.writerow([whole, part, jj, row[3], row[4], label, bin_label])
                if split == 'dev':
                    dvw.writerow([whole, part, jj, row[3], row[4], label, bin_label])
                if split == 'test':
                    tew.writerow([whole, part, jj, row[3], row[4], label, bin_label])

    trf.close()
    dvf.close()
    tef.close()
