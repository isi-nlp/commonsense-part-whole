"""
    Given a file of triples, predicts the labels given statistics from Google n-grams on the frequency of JJ amod PART
"""
import argparse, csv, operator
from collections import OrderedDict
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help="file of triples to predict")
    args = parser.parse_args()

    print("reading noun-adjective counts from google dependency n-grams...")
    with open('../../data/ngrams/nn-jj-amod/nn.jj.amod.all') as f:
        r = csv.reader(f, delimiter=' ')
        noun2jjs = {}
        for row in r:
            noun = row[0]
            jjs = OrderedDict()
            for i in range(1,len(row),2):
                jjs[row[i]] = int(row[i+1])
            noun2jjs[noun] = jjs

    with open(args.file) as f:
        triples = []
        preds = []
        for row in csv.reader(f):
            triples.append(tuple(row[:3]))
            part, jj = row[1], row[2]
            if part in noun2jjs:
                jj_order = [jj for jj, count in sorted(noun2jjs[part].items(), key=operator.itemgetter(1), reverse=True)]
                if jj in jj_order:
                    norm_rank = jj_order.index(jj) / len(jj_order)
                    #guess impossible by default (if never found)
                    pred = 1
                    for i in range(5,1,-1):
                        #use this exponential threshold thing, idk
                        thresh = np.exp2(-i)
                        if norm_rank < thresh:
                            pred = i
                            break
                else:
                    #impossible if not found in list
                    pred = 1
            else:
                #impossible if not found in list
                pred = 1
            preds.append(pred)
    import pdb; pdb.set_trace()
