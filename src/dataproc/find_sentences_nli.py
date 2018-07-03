"""
    Look thru all NLI datasets for valid sentences (context or hypothesis)
"""
import csv, json, os, re, sys, time
from multiprocessing import Pool

from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

def process_dset(fname):
    whole_jjs = {}
    with open(sys.argv[1]) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            whole_jjs[(row[0], row[2])] = set()

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
                    if (w2l, w1l) in whole_jjs:
                        whole_jjs[(w2l,w1l)].add(sent)
                    elif (w1l, w2l) in whole_jjs:
                        whole_jjs[(w1l,w2l)].add(sent)
                    else:
                        #also look for multi-word wholes
                        multi_part = ' '.join((w1l,w2l))
                        if ix + 2 <= len(words) - 1 and (multi_part, words[ix + 2].lower()) in whole_jjs:
                            #if validate(multi_part, words[ix + 2].lower(), sent):
                            whole_jjs[(multi_part, words[ix + 2].lower())].add(sent)
                        elif ix - 1 >= 0 and (multi_part, words[ix - 1].lower()) in whole_jjs:
                            #if validate(multi_part, words[ix - 1].lower(), sent):
                            whole_jjs[(multi_part, words[ix - 1].lower())].add(sent)
            if idx % 1000 == 0:
                print(fname, idx)
    return whole_jjs

if __name__ == "__main__":
    csv.field_size_limit(sys.maxsize)

    whole_jjs = {}
    with open(sys.argv[1]) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            whole_jjs[(row[0], row[2])] = set()

    dsets = ['/home/jamesm/commonsense-part-whole/data/snli_1.0/snli_1.0_train.txt', '/home/jamesm/commonsense-part-whole/data/multinli_0.9/multinli_0.9_train.txt']
    pool = Pool(2)
    tot_sentences = 0
    start = time.time()
    for ix, wjj in enumerate(pool.imap_unordered(process_dset, dsets)):
        for tup,sents in wjj.items():
            whole_jjs[tup].update(sents)

    with open('../../data/candidates/wj-nli-sentences.json', 'w') as of:
        #put whole-jj into a single string and do set->list so the json works
        wjj = {', '.join(tup):list(snts) for tup,snts in whole_jjs.items()}
        json.dump(wjj, of, indent=1)
