"""
    Look thru all downloaded gutenberg books for sentences with [adj whole]
"""
import csv, json, os, re, sys

from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

whole_jjs = {}
with open(sys.argv[1]) as f:
    r = csv.reader(f)
    #header
    next(r)
    for row in r:
        whole_jjs[(row[0], row[2])] = set()

BASE_DIR = '../../data/gutenberg/www.gutenberg.org/robot/'
sentences_found = 0
num_books = 0
for f in tqdm(os.listdir(BASE_DIR)):
    if f.endswith('.txt'):
        book = open('%s/%s' % (BASE_DIR, f)).read().replace('\n', ' ')
        #get rid of multiple whitespace
        book = re.sub('\s+', ' ', book)
        sents = sent_tokenize(book)
        for sent in sents:
            words = word_tokenize(sent)
            for w1, w2 in zip(words[:-1], words[1:]):
                w1l, w2l = w1.lower(), w2.lower()
                #consider both possiblities - red car and car red. just cause
                if (w2l, w1l) in whole_jjs:
                    whole_jjs[(w2l,w1l)].add(sent)
                    sentences_found += 1
                    if sentences_found % 100 == 0:
                        print("%d sentences found" % sentences_found)
                        wjjs_with_context = len([tup for tup,snt in whole_jjs.items() if len(snt) > 0])
                        print("%d whole-jjs with sentences" % wjjs_with_context)
                elif  (w1l, w2l) in whole_jjs:
                    whole_jjs[(w1l,w2l)].add(sent)
                    sentences_found += 1
                    if sentences_found % 100 == 0:
                        print("%d sentences found" % sentences_found)
                        wjjs_with_context = len([tup for tup,snt in whole_jjs.items() if len(snt) > 0])
                        print("%d whole-jjs with sentences" % wjjs_with_context)
        num_books += 1
        #just write periodically book idc
        if num_books % 10 == 0:
            with open('../../data/candidates/wj-gutenberg-sentences.json', 'w') as of:
                #put whole-jj into a single string and do set->list so the json works
                wjj = {', '.join(tup):list(snts) for tup,snts in whole_jjs.items()}
                json.dump(wjj, of, indent=1)
 
