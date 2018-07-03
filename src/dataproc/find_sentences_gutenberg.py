"""
    Look thru all downloaded gutenberg books for sentences with [adj whole]
"""
import csv, json, os, re, sys, time
from multiprocessing import Pool

from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from tqdm import tqdm

nlp = spacy.load('en')

def validate(whole, adj, sent):
    #tag sentences
    doc = nlp(sent)
    for tok in doc:
        if tok.text.lower() == adj:
            head = tok.head.text.lower()
            if tok.dep_ == 'amod' and head == whole or (' ' in whole and whole.split()[0] == head or whole.split()[1] == head):
                return True
    return False

def process_book(fname):
    whole_jjs = {}
    with open(sys.argv[1]) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            whole_jjs[(row[0], row[2])] = set()

    book = open('%s/%s' % (BASE_DIR, fname)).read().replace('\n', ' ')
    #get rid of multiple whitespace
    book = re.sub('\s+', ' ', book)
    sents = sent_tokenize(book)
    sentences_found = 0
    for sent in sents:
        words = word_tokenize(sent)
        for ix,(w1, w2) in enumerate(zip(words[:-1], words[1:])):
            w1l, w2l = w1.lower(), w2.lower()
            #consider both possiblities - red car and car red. just cause
            if (w2l, w1l) in whole_jjs:
                #if validate(w2l, w1l, sent):
                whole_jjs[(w2l,w1l)].add(sent)
                sentences_found += 1
            elif (w1l, w2l) in whole_jjs:
                #if validate(w1l, w2l, sent):
                whole_jjs[(w1l,w2l)].add(sent)
                sentences_found += 1
            else:
                #also look for multi-word wholes
                multi_part = ' '.join((w1l,w2l))
                if ix + 2 <= len(words) - 1 and (multi_part, words[ix + 2].lower()) in whole_jjs:
                    #if validate(multi_part, words[ix + 2].lower(), sent):
                    whole_jjs[(multi_part, words[ix + 2].lower())].add(sent)
                elif ix - 1 >= 0 and (multi_part, words[ix - 1].lower()) in whole_jjs:
                    #if validate(multi_part, words[ix - 1].lower(), sent):
                    whole_jjs[(multi_part, words[ix - 1].lower())].add(sent)
    return whole_jjs, sentences_found

if __name__ == "__main__":
    BASE_DIR = '/home/jamesm/e/robot/'

    whole_jjs = {}
    with open(sys.argv[1]) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            whole_jjs[(row[0], row[2])] = set()

    books = [fname for fname in os.listdir(BASE_DIR) if fname.endswith('.txt')]
    pool = Pool(processes=12)
    tot_sentences = 0
    start = time.time()
    for ix, (wjj, sentences_found) in enumerate(pool.imap_unordered(process_book, books)):
        for tup,sents in wjj.items():
            whole_jjs[tup].update(sents)

        tot_sentences += sentences_found
        if tot_sentences % 100 == 0:
            print("%d sentences found" % tot_sentences)
            wjjs_with_context = len([tup for tup,snt in whole_jjs.items() if len(snt) > 0])
            print("%d whole-jjs with sentences" % wjjs_with_context)

        if ix % 100 == 0:
            print("processed %d books. writing..." % ix,)
            #just write periodically book idc
            with open('../../data/candidates/wj-gutenberg-sentences.json', 'w') as of:
                #put whole-jj into a single string and do set->list so the json works
                wjj = {', '.join(tup):list(snts) for tup,snts in whole_jjs.items()}
                json.dump(wjj, of, indent=1)
            print("done")
            elapsed = time.time() - start
            print("elapsed time: %f" % (elapsed))
            print("books per second: %f" % ((ix + 1) / elapsed))

    with open('../../data/candidates/wj-gutenberg-sentences.json', 'w') as of:
        #put whole-jj into a single string and do set->list so the json works
        wjj = {', '.join(tup):list(snts) for tup,snts in whole_jjs.items()}
        json.dump(wjj, of, indent=1)
