"""
    Read snli style sentence data for each part-whole-adjective. 
    Copy labels and splits from that that exists for triple datasets
"""
import argparse, csv, json, random
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
    trw.writerow(['whole', 'part', 'jj', 'hypothesis', 'hsrc', 'context', 'csrc', 'label', 'bin_label'])
    dvw = csv.writer(dvf, delimiter='\t')
    dvw.writerow(['whole', 'part', 'jj', 'hypothesis', 'hsrc', 'context', 'csrc', 'label', 'bin_label'])
    tew = csv.writer(tef, delimiter='\t')
    tew.writerow(['whole', 'part', 'jj', 'hypothesis', 'hsrc', 'context', 'csrc', 'label', 'bin_label'])
    
    with open(args.file) as f:
        trip2sentpairs = json.load(f)
        for trip, (split, label, bin_label) in trip2labels.items():
            whole, part, jj = trip
            joined = ','.join(trip)
            if joined in trip2sentpairs:
                for sent1, sent2 in trip2sentpairs[joined]:
                    if split == 'train':
                        trw.writerow([whole, part, jj, sent1[0], sent1[1], sent2[0], sent2[1], label, bin_label])
                    if split == 'dev':
                        dvw.writerow([whole, part, jj, sent1[0], sent1[1], sent2[0], sent2[1], label, bin_label])
                    if split == 'test':
                        tew.writerow([whole, part, jj, sent1[0], sent1[1], sent2[0], sent2[1], label, bin_label])
            else:
                #I think these are WJ's missing because we didn't find sentences for them?
                det = 'an' if jj[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
                sent1 = f"There is {det} {jj} {whole}."
                sent2 = f"The {whole}'s {part} is {jj}."
                if split == 'train':
                    trw.writerow([whole, part, jj, sent1, 'syn', sent2, 'syn', label, bin_label])
                if split == 'dev':
                    dvw.writerow([whole, part, jj, sent1, 'syn', sent2, 'syn', label, bin_label])
                if split == 'test':
                    tew.writerow([whole, part, jj, sent1, 'syn', sent2, 'syn', label, bin_label])

                
        #for trip, sentpairs in trip2sentpairs.items():
        #    trip = tuple(trip.split(','))
        #    whole, part, jj = trip
        #    if trip in trip2labels:
        #        split, label, bin_label = trip2labels[trip]
        #        for premise, hypothesis in sentpairs:
        #            if split == 'train':
        #                trw.writerow([whole, part, jj, premise, hypothesis, label, bin_label])
        #            if split == 'dev':
        #                dvw.writerow([whole, part, jj, premise, hypothesis, label, bin_label])
        #            if split == 'test':
        #                tew.writerow([whole, part, jj, premise, hypothesis, label, bin_label])
        #    else:
        #        if ' ' in whole:
        #            print(trip)

        #r = csv.reader(f, delimiter='\t')
        ##header
        #next(r)
        #for row in r:
        #    whole, part, jj = tuple(row[:3])
        #    if (whole, part, jj) in trip2labels:
        #        split, label, bin_label = trip2labels[(whole, part, jj)]
        #        if split == 'train':
        #            trw.writerow([whole, part, jj, row[3], row[4], label, bin_label])
        #        if split == 'dev':
        #            dvw.writerow([whole, part, jj, row[3], row[4], label, bin_label])
        #        if split == 'test':
        #            tew.writerow([whole, part, jj, row[3], row[4], label, bin_label])

    trf.close()
    dvf.close()
    tef.close()
