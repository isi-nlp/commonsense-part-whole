"""
    Given gigaword, gutenberg, snli input, combine results of all.
    Select (1? 2-3?) sentences per whole-adj, combine with manufactured part-adj hypothesis.
"""
import csv
import json
import os
from collections import defaultdict, namedtuple
from multiprocessing import Pool
from random import shuffle

import spacy
from tqdm import tqdm

nlp = spacy.load('en')

def validate(whole, adj, sent):
    #length constraints
    if len(sent.split()) >= 50 or len(sent.split()) <= 5:
        return False
    #tag sentences
    doc = nlp(sent, disable=['ner'])
    for tok in doc:
        if tok.text.lower() == adj:
            head = tok.head
            #check POS tags
            if tok.pos_ == 'ADJ' and head.pos_ == 'NOUN':
                try:
                    #check arc
                    if tok.dep_ == 'amod':
                        if head.text.lower() == whole:
                            return True
                        elif ' ' in whole:
                            return whole.split()[0] == head.text.lower() or whole.split()[1] == head.text.lower()
                except:
                    import pdb; pdb.set_trace()
    return False

if __name__ == "__main__":
    sentsource = namedtuple('sentsource', ['sentence', 'source'])

    print("reading captions")
    wjjs = defaultdict(set)
    with open('../../data/candidates/wj-caption-sentences.json') as f:
        wjs = json.load(f)
        for tup,sents in wjs.items():
            tup = tuple(tup.split(', '))
            wjjs[tup] = [sentsource(sent, 'caption') for sent in set(sents)]

    print("reading gutenberg")
    with open('../../data/candidates/wj-gutenberg-sentences.json') as f:
        wjs = json.load(f)
        for tup,sents in wjs.items():
            tup = tuple(tup.split(', '))
            wjjs[tup].extend([sentsource(sent, 'gutenberg') for sent in set(sents)])

    print("reading gigaword")
    with open('../../data/candidates/wj-gigaword-sentences.json') as f:
        wjs = json.load(f)
        for tup,sents in wjs.items():
            tup = tuple(tup.split(', '))
            wjjs[tup].extend([sentsource(sent, 'gigaword') for sent in set(sents)])

    print("reading nli")
    with open('../../data/candidates/wj-nli-sentences.json') as f:
        wjs = json.load(f)
        for tup,sents in wjs.items():
            tup = tuple(tup.split(', '))
            wjjs[tup].extend([sentsource(sent, 'nli') for sent in set(sents)])

    print("reading part whole relations")
    whole2parts = defaultdict(set)
    with open('../../data/candidates/vis_uniq_less_colors.csv') as f:
        r = csv.reader(f)
        for row in r:
            whole2parts[row[0]].add(row[1])


    def get_lines(item):
        tup, sentsources = item
        if len(sentsources) == 0:
            return []
        #select up to 250 at random
        shuffle(sentsources)
        sentsources = sentsources[:250]

        #filter selected sentences
        sentences = [ss.sentence for ss in sentsources if validate(*tup, ss.sentence)]

        #select up to 142 at random (max. number of parts)
        shuffle(sentences)
        sents = sentences[:142]
        whole, jj = tup
        rows = []
        if len(sents) > 0:
            #write one line and one sentence per part
            for i,part in enumerate(whole2parts[whole]):
                hypothesis = "The {}'s {} is {}.".format(whole, part, jj)
                #context sentence for a part is the next one, unless there are more parts than sentences, in which case repeat the last one
                rows.append([whole, part, jj, hypothesis, sents[min(i,len(sents)-1)]])
        return rows

    to_write = []
    pool = Pool(processes=os.cpu_count())
    for rows in pool.imap_unordered(get_lines, wjjs.items()):
        to_write.extend(rows)
        if len(to_write) // 100 != (len(to_write) - len(rows)) // 100:
            print(len(to_write))

    with open('../../data/candidates/snli-style.csv', 'w') as of:
        w = csv.writer(of, delimiter='\t')
        header = ['whole', 'part', 'adj', 'hypothesis', 'context']
        #header.extend(['context{}'.format(i+1) for i in range(10)])
        w.writerow(header)
        for line in to_write:
            w.writerow(line)
        
