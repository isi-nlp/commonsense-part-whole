"""
    Look thru all COCO captions for sentences
"""
import argparse, csv, json, os, re, sys, time
from multiprocessing import Pool

from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="triples file")
    parser.add_argument("noun", choices=['part', 'whole'], help="look for whole-jj or part-jj?")
    args = parser.parse_args()

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
 
    for fold in ['train', 'val']:
        annotations = json.load(open('../../data/coco/annotations/captions_%s2017.json' % fold))
        caps = annotations['annotations']
        for cap in tqdm(caps):
            caption = cap['caption']
            words = word_tokenize(caption)
            for ix,(w1, w2) in enumerate(zip(words[:-1], words[1:])):
                w1l, w2l = w1.lower(), w2.lower()
                #consider both possiblities - red car and car red. just cause
                if (w2l, w1l) in noun_jjs:
                    noun_jjs[(w2l,w1l)].add(caption)
                elif (w1l, w2l) in noun_jjs:
                    noun_jjs[(w1l,w2l)].add(caption)
                else:
                    #also look for multi-word nouns
                    multi_part = ' '.join((w1l,w2l))
                    if ix + 2 <= len(words) - 1 and (multi_part, words[ix + 2].lower()) in noun_jjs:
                        #if validate(multi_part, words[ix + 2].lower(), caption):
                        noun_jjs[(multi_part, words[ix + 2].lower())].add(caption)
                    elif ix - 1 >= 0 and (multi_part, words[ix - 1].lower()) in noun_jjs:
                        #if validate(multi_part, words[ix - 1].lower(), caption):
                        noun_jjs[(multi_part, words[ix - 1].lower())].add(caption)

    name = 'wj' if args.noun == 'whole' else 'pj'
    with open(f'../../data/candidates/{name}-caption-sentences.json', 'w') as of:
        #put noun-jj into a single string and do set->list so the json works
        njj = {', '.join(tup):list(snts) for tup,snts in noun_jjs.items()}
        json.dump(njj, of, indent=1)
