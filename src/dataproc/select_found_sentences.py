"""
    Given gigaword, gutenberg, snli input, combine results of all.
    Select (1? 2-3?) sentences per whole-adj, combine with retrieved or manufactured part-adj hypotheses.
"""
import csv
import json
import os
import sys
import time
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
                return s + ' .'
    return None

if __name__ == "__main__":
    if not sys.argv[1].isnumeric():
        print(f"usage: python select_found_sentences.py [cpu count]")
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
                        pwsents[(whole, part)].add(valid_sent)

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
    pw2jjs = defaultdict(set)
    with open('../../data/annotated/full_all.csv') as f:
        r = csv.reader(f)
        for row in r:
            whole2parts[row[0]].add(row[1])
            pw2jjs[(row[0], row[1])].add(row[2])

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
        trip2sentpairs = defaultdict(list)

        #for each triple, write up to 5 natural sentence pairs. Use manufactured if none available
        contexts_used = 0
        for part in whole2parts[whole]:
            hypotheses = list(pwsents[(whole, part)])
            if len(hypotheses) > 0:
                shuffle(hypotheses)
                for i, hyp in enumerate(hypotheses[:5]):

                    if len(sents) > 0:
                        context = sents[min(contexts_used, len(sents)-1)]
                    else:
                        det = 'an' if jj[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
                        context = f"There is {det} {jj} {whole}."

                    trip2sentpairs[','.join([whole, part, jj])].append([context, hyp])
                    contexts_used += 1
            else:
                hyp = "The {}'s {} is {}.".format(whole, part, jj)

                if len(sents) > 0:
                    context = sents[min(contexts_used, len(sents)-1)]
                else:
                    det = 'an' if jj[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
                    context = f"There is {det} {jj} {whole}."

                #context sentence for a part is the next one, unless there are more parts than sentences, in which case repeat the last one
                trip2sentpairs[','.join([whole, part, jj])].append([context, hyp])
                contexts_used += 1
        return trip2sentpairs

    trip2sentpairs = defaultdict(list)
    print(f"num wjs: {len(wjjs)}")
    #num_pairs = 0
    pool = Pool(processes=int(sys.argv[1]))
    for num_wjjs,dct in enumerate(pool.imap_unordered(get_lines, wjjs.items())):
        #num_pairs_batch = 0
        if type(dct) is list:
            continue
            #import pdb; pdb.set_trace()
        for trip, sentpairs in dct.items():
            trip2sentpairs[trip].extend(sentpairs)
            #num_pairs += len(sentpairs)
            #num_pairs_batch += len(sentpairs)
        #if num_pairs // 100 != (num_pairs - num_pairs_batch) // 100:
        if num_wjjs % 100 == 0:
            print(num_wjjs)
            #save in case we get interrupted
        #if num_pairs // 10000 != (num_pairs - num_pairs_batch) // 10000:
            with open('../../data/candidates/trip2sentpairs.json', 'w') as of:
                json.dump(trip2sentpairs, of)

    with open('../../data/candidates/trip2sentpairs.json', 'w') as of:
        json.dump(trip2sentpairs, of)
    #with open('../../data/candidates/snli-style.csv', 'w') as of:
    #    w = csv.writer(of, delimiter='\t')
    #    header = ['whole', 'part', 'adj', 'hypothesis', 'context']
    #    #header.extend(['context{}'.format(i+1) for i in range(10)])
    #    w.writerow(header)
    #    for line in to_write:
    #        w.writerow(line)
        
