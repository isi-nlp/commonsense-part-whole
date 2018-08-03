"""
    Look thru all NLI datasets for valid sentences (context or hypothesis)
"""
import argparse, csv, json, os, re, sys, time
from multiprocessing import Pool

from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

def process_dset(fname):
    noun_jjs = {}
    with open(sys.argv[1]) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            if args.noun == 'part':
                noun_jjs[(row[1], row[2])] = set()
            else:
                noun_jjs[(row[0], row[2])] = set()

    with open(fname) as f:
        r = csv.reader(f, delimiter='\t')
        #header
        next(r)
        for idx,row in enumerate(r):
            context, hyp = row[5:7]
            sents = [context, hyp]
            for sent in sents:
                words = word_tokenize(sent)
                for ix,(w1, w2) in enumerate(zip(words[:-1], words[1:])):
                    w1l, w2l = w1.lower(), w2.lower()
                    #consider both possiblities - red car and car red. just cause
                    if (w2l, w1l) in noun_jjs:
                        noun_jjs[(w2l,w1l)].add(sent)
                    elif (w1l, w2l) in noun_jjs:
                        noun_jjs[(w1l,w2l)].add(sent)
                    else:
                        #also look for multi-word nouns
                        multi_part = ' '.join((w1l,w2l))
                        if ix + 2 <= len(words) - 1 and (multi_part, words[ix + 2].lower()) in noun_jjs:
                            #if validate(multi_part, words[ix + 2].lower(), sent):
                            noun_jjs[(multi_part, words[ix + 2].lower())].add(sent)
                        elif ix - 1 >= 0 and (multi_part, words[ix - 1].lower()) in noun_jjs:
                            #if validate(multi_part, words[ix - 1].lower(), sent):
                            noun_jjs[(multi_part, words[ix - 1].lower())].add(sent)
            if idx % 1000 == 0:
                print(fname, idx)
    return noun_jjs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="triples file")
    parser.add_argument("cpu_count", type=int, help="num cpus to use")
    parser.add_argument("noun", choices=['part', 'whole'], help="look for whole-jj or part-jj?")
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)

    noun_jjs = {}
    with open(args.file) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            if args.noun == 'part':
                noun_jjs[(row[1], row[2])] = set()
            else:
                noun_jjs[(row[0], row[2])] = set()

    dsets = ['../../data/snli_1.0/snli_1.0_train.txt', '../../data/multinli_0.9/multinli_0.9_train.txt']
    pool = Pool(2)
    tot_sentences = 0
    start = time.time()
    for ix, njj in enumerate(pool.imap_unordered(process_dset, dsets)):
        for tup,sents in njj.items():
            noun_jjs[tup].update(sents)

    name = 'wj' if args.noun == 'whole' else 'pj'
    with open(f'../../data/candidates/{name}-nli-sentences.json', 'w') as of:
        #put noun-jj into a single string and do set->list so the json works
        njj = {', '.join(tup):list(snts) for tup,snts in noun_jjs.items()}
        json.dump(njj, of, indent=1)
