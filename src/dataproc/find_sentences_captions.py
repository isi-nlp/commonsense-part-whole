"""
    Look thru all COCO captions for sentences
"""
import csv, json, os, re, sys, time
from multiprocessing import Pool

from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

if __name__ == "__main__":
    whole_jjs = {}
    with open(sys.argv[1]) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            whole_jjs[(row[0], row[2])] = set()
 
    for fold in ['train', 'val']:
        annotations = json.load(open('../../data/coco/annotations/captions_%s2017.json' % fold))
        caps = annotations['annotations']
        for cap in tqdm(caps):
            caption = cap['caption']
            words = word_tokenize(caption)
            for ix,(w1, w2) in enumerate(zip(words[:-1], words[1:])):
                w1l, w2l = w1.lower(), w2.lower()
                #consider both possiblities - red car and car red. just cause
                if (w2l, w1l) in whole_jjs:
                    whole_jjs[(w2l,w1l)].add(caption)
                elif (w1l, w2l) in whole_jjs:
                    whole_jjs[(w1l,w2l)].add(caption)
                else:
                    #also look for multi-word wholes
                    multi_part = ' '.join((w1l,w2l))
                    if ix + 2 <= len(words) - 1 and (multi_part, words[ix + 2].lower()) in whole_jjs:
                        #if validate(multi_part, words[ix + 2].lower(), caption):
                        whole_jjs[(multi_part, words[ix + 2].lower())].add(caption)
                    elif ix - 1 >= 0 and (multi_part, words[ix - 1].lower()) in whole_jjs:
                        #if validate(multi_part, words[ix - 1].lower(), caption):
                        whole_jjs[(multi_part, words[ix - 1].lower())].add(caption)

    with open('../../data/candidates/wj-caption-sentences.json', 'w') as of:
        #put whole-jj into a single string and do set->list so the json works
        wjj = {', '.join(tup):list(snts) for tup,snts in whole_jjs.items()}
        json.dump(wjj, of, indent=1)
