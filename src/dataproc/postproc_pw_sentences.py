import argparse
from collections import defaultdict
import json
import os
import sys
import time
from multiprocessing import Pool

import spacy
from spacy.tokens import Doc
from tqdm import tqdm

#super dumb stuff needed to simply split on whitespace when using spacy
class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

def get_of_sents(fn):
    start = time.time()
    pw2sents = defaultdict(list)
    for i,line in enumerate(open(fn)):
        sent = nlp(line.strip())
        if i % 1000 == 0:
            print(f'line: {i}, time: {time.time() - start}')
        for i,tok in enumerate(sent):
            if tok.dep_ == 'pobj':
                if tok.text in wholes:
                    whole = tok
                    if tok.head.text == 'of':
                        part = tok.head.head
                        if part.text in parts:
                            if (whole.text, part.text) in pws:
                                pw2sents[','.join((whole.text, part.text))].append(sent.text)
                elif i > 0:
                    #multi-word wholes
                    whole = ' '.join((sent[i-1].text, tok.text))
                    if whole in wholes:
                        if tok.head.text == 'of':
                            part = tok.head.head
                            if part.text in parts and (whole, part.text) in pws:
                                pw2sents[','.join((whole, part.text))].append(sent.text)
    return pw2sents, fn

def get_poss_sents(fn):
    start = time.time()
    pw2sents = defaultdict(list)
    for i,line in enumerate(open(fn)):
        sent = nlp(line.strip())
        if i % 1000 == 0:
            print(f'line: {i}, time: {time.time() - start}')
        for tok in sent:
            if tok.text == "'s":
                whole = tok.head
                if whole.dep_ == 'poss':
                    if whole.text in wholes:
                        part = whole.head
                        if part.text in parts and (whole.text, part.text) in pws:
                            pw2sents[','.join((whole.text, part.text))].append(sent.text)
                    elif whole.i > 0:
                        #multi-word wholes
                        whole = ' '.join((sent[whole.i-1].text, whole.text))
                        if whole in wholes:
                            part = tok.head.head
                            if part.text in parts and (whole, part.text) in pws:
                                pw2sents[','.join((whole, part.text))].append(sent.text)
    return pw2sents, fn

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=['of', 'poss'])
parser.add_argument("data", choices=['umbc', 'giga'])
parser.add_argument("processes", type=int)
args = parser.parse_args()

nlp = spacy.load('en', disable=['ner', 'textcat'])
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

HOME = '/auto/nlg-05/jgm_234/'
DATA = os.path.join(HOME, 'commonsense-part-whole/data/')
PATH = os.path.join(DATA, 'gigaword5-treebank' if args.data == 'giga' else 'umbc-webbased-parsed')

wholes = set([line.strip().split(',')[0] for line in open(os.path.join(DATA, 'annotated/full_all.csv'))])
parts = set([line.strip().split(',')[1] for line in open(os.path.join(DATA, 'annotated/full_all.csv'))])
pws = set([(line.strip().split(',')[0], line.strip().split(',')[1]) for line in open(os.path.join(DATA, 'annotated/full_all.csv'))])

pw2sents = defaultdict(list)
files = [os.path.join(PATH, d) for d in os.listdir(PATH) if d.endswith(f'{args.mode}.pwsents')]

file_fn = get_of_sents if args.mode == 'of' else get_poss_sents

pool = Pool(processes=args.processes)
fdone = []
for ix,(res, fn) in enumerate(pool.imap_unordered(file_fn, files)):
    for pw, sents in res.items():
        pw2sents[pw].extend(sents)
    #pw2sents.update(res)
    fdone.append(fn)
    #save in case we get interrupted
    if ix % 1 == 0:
        with open(f'{args.mode}.{args.data}.sents.json', 'w') as of:
            json.dump(dict(pw2sents), of)
        with open(f'{args.mode}.{args.data}.filesdone.txt', 'w') as of:
            for fd in fdone:
                of.write(fd + '\n')

with open(f'{args.mode}.{args.data}.sents.json', 'w') as of:
    json.dump(dict(pw2sents), of)

with open(f'{args.mode}.{args.data}.filesdone.txt', 'w') as of:
    for fd in fdone:
        of.write(fd + '\n')
