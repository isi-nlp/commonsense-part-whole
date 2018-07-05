"""
    Given gigaword, gutenberg, snli input, combine results of all.
    Select (1? 2-3?) sentences per whole-adj, combine with manufactured part-adj hypothesis.
"""
import json
from collections import defaultdict, namedtuple

import spacy

nlp = spacy.load('en')

def validate(whole, adj, sent):
    #length constraints
    if len(sent.split()) >= 50 or len(sent.split()) <= 5:
        return False
    #tag sentences
    doc = nlp(sent, disable=['ner'])
    for tok in doc:
        if tok.text.lower() == adj:
            head = tok.head.text.lower()
            #check POS tags
            if tok.pos_ == 'JJ' and head.pos_ == 'NN':
                #check arc
                if tok.dep_ == 'amod' and head == whole or (' ' in whole and whole.split()[0] == head or whole.split()[1] == head):
                    return True
    return False

if __name__ == "__main__":
    sentsource = namedtuple('sentsource', ['sentence', 'source'])

    wjjs = defaultdict(set)
    with open('../../data/candidates/wj-gutenberg-sentences.json') as f:
        wjs = json.load(f)
        for tup,sents in wjs.items():
            tup = tuple(tup.split(', '))
            wjjs[tup] = [sentsource(sent, 'gutenberg') for sent in set(sents)]

    with open('../../data/candidates/wj-gigaword-sentences.json') as f:
        wjs = json.load(f)
        for tup,sents in wjs.items():
            tup = tuple(tup.split(', '))
            wjjs[tup].extend([sentsource(sent, 'gigaword') for sent in set(sents)])

    with open('../../data/candidates/wj-nli-sentences.json') as f:
        wjs = json.load(f)
        for tup,sents in wjs.items():
            tup = tuple(tup.split(', '))
            wjjs[tup].extend([sentsource(sent, 'nli') for sent in set(sents)])

    with open('../../data/candidates/wj-caption-sentences.json') as f:
        wjs = json.load(f)
        for tup,sents in wjs.items():
            tup = tuple(tup.split(', '))
            wjjs[tup].extend([sentsource(sent, 'caption') for sent in set(sents)])

    with open('../../data/candidates/snli-style.csv', 'w') as of:
        w = csv.writer(of)
        w.writerow(['whole', 'part', 'adj', 'hypothesis', 'context1', 'context2', 'context3', 'context4', 'context5'])
        for tup, sentsources in wjjs.items():
            #filter sentences
            sentences = [ss.sentences for ss in sentsources if validate(*tup, ss.sentence)]
            #now select up to 5 at random

