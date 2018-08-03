"""
    Given gigaword, gutenberg, snli input, combine results of all.
    Select up to 5 sentences per triple, combine with retrieved or manufactured part-adj hypotheses.
"""
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict, namedtuple
from multiprocessing import Pool
from random import shuffle

import spacy
from tqdm import tqdm

nlp = spacy.load('en')

def validate(noun, adj, sent):
    #length constraints
    if len(sent.split()) >= 50 or len(sent.split()) <= 5:
        return False
    #tag sentences
    doc = nlp(sent, disable=['ner', 'textcat'])
    for tok in doc:
        if tok.text.lower() == adj:
            head = tok.head
            #check POS tags
            if tok.pos_ == 'ADJ' and head.pos_ == 'NOUN':
                try:
                    #check arc
                    if tok.dep_ == 'amod':
                        if head.text.lower() == noun:
                            return True
                        elif ' ' in noun:
                            return noun.split()[0] == head.text.lower() or noun.split()[1] == head.text.lower()
                except:
                    import pdb; pdb.set_trace()
    return False

def extract_sent(part, whole, sent):
    for s in sent.split(' . '):
        part_found = False
        whole_found = False
        for word in s.split():
            if word.lower() == whole:
                whole_found = True
            if word.lower() == part:
                part_found = True
            if whole_found and part_found:
                #add back the period
                s = replace_special_tokens(s)
                return s + ' .'
    return None

def replace_special_tokens(s):
    return s.replace('-LRB-', '(')\
            .replace('-RRB-', ')')\
            .replace('-LSB-', '[')\
            .replace('-RSB-', ']')\
            .replace('-LCB-', '{')\
            .replace('-RCB-', '}')


if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].isnumeric():
        print(f"usage: python select_found_sentences.py [cpu count]")
        sys.exit(0)
    sentsource = namedtuple('sentsource', ['sentence', 'source'])

    print("reading pw sentences")
    pwsents = defaultdict(set)
    with open('../../data/candidates/pw_sents.json') as f:
        obj = json.load(f)
        for whole, dct in obj.items():
            for part, sents in dct.items():
                #filter out extra sentences b/c there are often several for some reason
                for sent in sents:
                    valid_sent = extract_sent(part, whole, sent)
                    if valid_sent is not None:
                        pwsents[(whole, part)].add(sentsource(valid_sent, 'pw'))

    print("reading captions")
    wjjs = defaultdict(set)
    with open('../../data/candidates/wj-caption-sentences.json') as f:
        wjs = json.load(f)
        for tup,sents in wjs.items():
            tup = tuple(tup.split(', '))
            wjjs[tup] = [sentsource(sent, 'caption-wj') for sent in set(sents)]

    print("reading gutenberg")
    with open('../../data/candidates/wj-gutenberg-sentences.json') as f:
        wjs = json.load(f)
        for tup,sents in wjs.items():
            tup = tuple(tup.split(', '))
            wjjs[tup].extend([sentsource(sent, 'gutenberg-wj') for sent in set(sents)])

    print("reading gigaword")
    with open('../../data/candidates/wj-gigaword-sentences.json') as f:
        wjs = json.load(f)
        for tup,sents in wjs.items():
            tup = tuple(tup.split(', '))
            wjjs[tup].extend([sentsource(sent, 'gigaword-wj') for sent in set(sents)])

    print("reading nli")
    with open('../../data/candidates/wj-nli-sentences.json') as f:
        wjs = json.load(f)
        for tup,sents in wjs.items():
            tup = tuple(tup.split(', '))
            wjjs[tup].extend([sentsource(sent, 'nli-wj') for sent in set(sents)])

    print("reading captions")
    pjjs = defaultdict(set)
    with open('../../data/candidates/pj-caption-sentences.json') as f:
        pjs = json.load(f)
        for tup,sents in pjs.items():
            tup = tuple(tup.split(', '))
            pjjs[tup] = [sentsource(sent, 'caption-pj') for sent in set(sents)]

    print("reading gutenberg")
    with open('../../data/candidates/pj-gutenberg-sentences.json') as f:
        pjs = json.load(f)
        for tup,sents in pjs.items():
            tup = tuple(tup.split(', '))
            pjjs[tup].extend([sentsource(sent, 'gutenberg-pj') for sent in set(sents)])

    print("reading gigaword")
    with open('../../data/candidates/pj-gigaword-sentences.json') as f:
        pjs = json.load(f)
        for tup,sents in pjs.items():
            tup = tuple(tup.split(', '))
            pjjs[tup].extend([sentsource(sent, 'gigaword-pj') for sent in set(sents)])

    print("reading nli")
    with open('../../data/candidates/pj-nli-sentences.json') as f:
        pjs = json.load(f)
        for tup,sents in pjs.items():
            tup = tuple(tup.split(', '))
            pjjs[tup].extend([sentsource(sent, 'nli-pj') for sent in set(sents)])


    print("reading part whole relations")
    whole2parts = defaultdict(set)
    pw2jjs = defaultdict(set)
    with open('../../data/annotated/full_all.csv') as f:
        r = csv.reader(f)
        for row in r:
            whole2parts[row[0]].add(row[1])
            pw2jjs[(row[0], row[1])].add(row[2])

    def get_lines(item):
        tup, jjs = item
        whole, part = tup
        trip2sentpairs = defaultdict(list)

        #for each triple, write up to 5 natural sentence pairs. Use manufactured if none available
        contexts_used = 0
        for jj in jjs:
            wjss = list(wjjs[(whole, jj)])
            pwss = list(pwsents[(whole, part)])
            pjss = list(pjjs[(part, jj)])
            shuffle(wjss)
            shuffle(pwss)
            shuffle(pjss)
            wjss = [ss for ss in wjss[:20] if validate(whole, jj, ss.sentence)]
            pwss = pwss[:5]
            pjss = [ss for ss in pjss[:20] if validate(part, jj, ss.sentence)]
            if len(wjss) > 0:
                if len(pwss) > 0:
                    #WJ+PW
                    sent1s = wjss[:5]
                    sent2s = pwss[:5]
                else:
                    if len(pjss) > 0:
                        #WJ+PJ
                        sent1s = wjss[:5]
                        sent2s = pjss[:5]
                    else:
                        #no PW or PJ, so replace the W in WJ with P
                        sent1s = wjss[:1]
                        sent2s = [sentsource(sent1s[0].sentence.replace(whole, part), 'replace')]
            else:
                #no WJ, do we have PW and PJ?
                if len(pjss) > 0 and len(pwss) > 0:
                    #write these
                    sent1s = pjss[:5]
                    sent2s = pwss[:5]
                elif len(pwss) > 0:
                    #no pj - do a fake wj
                    det = 'an' if whole[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
                    sent1s = [sentsource(f"There is {det} {jj} {whole}.", 'syn')]
                    sent2s = pwss[:1]
                elif len(pjss) > 0:
                    #no pw - do a fake wj
                    det = 'an' if whole[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
                    sent1s = [sentsource(f"There is {det} {jj} {whole}.", 'syn')]
                    sent2s = pjss[:1]
                else:
                    #nothing, double fake
                    det = 'an' if whole[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
                    sent1s = [sentsource(f"There is {det} {jj} {whole}.", 'syn')]
                    sent2s = [sentsource(f"The {part} is {jj}.", 'syn')]

            for sent1, sent2 in zip(sent1s, sent2s):
                #random ordering
                if random.random() > 0.5:
                    trip2sentpairs[','.join([whole, part, jj])].append([sent1, sent2])
                else:
                    trip2sentpairs[','.join([whole, part, jj])].append([sent2, sent1])
                
        return trip2sentpairs

    trip2sentpairs = defaultdict(list)
    print(f"len pw2jjs: {len(pw2jjs)}")
    pool = Pool(processes=int(sys.argv[1]))
    for num_done,dct in tqdm(enumerate(pool.imap_unordered(get_lines, pw2jjs.items()))):
        if type(dct) is list:
            continue
        for trip, sentpairs in dct.items():
            trip2sentpairs[trip].extend(sentpairs)
        if num_done % 100 == 0:
            #save in case we get interrupted
            with open('../../data/candidates/trip2sentpairs.json', 'w') as of:
                json.dump(trip2sentpairs, of)

    with open('../../data/candidates/trip2sentpairs.json', 'w') as of:
        json.dump(trip2sentpairs, of)
