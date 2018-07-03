"""
    Given gigaword, gutenberg, snli input, combine results of all.
    Select (1? 2-3?) sentences per whole-adj, combine with manufactured part-adj hypothesis.
"""
import json

wj_gut = {}
with open('/../../data/candidates/wj-gutenberg-sentences.json') as f:
    wjs = json.load(f)
    for tup,sents in wjs.items():
        tup = tuple(tup.split(', '))
        wj_gut[tup] = set(sents)

wj_gig = {}
with open('/../../data/candidates/wj-gigaword-sentences.json') as f:
    wjs = json.load(f)
    for tup,sents in wjs.items():
        tup = tuple(tup.split(', '))
        wj_gig[tup] = set(sents)

wj_nli = {}
with open('/../../data/candidates/wj-nli-sentences.json') as f:
    wjs = json.load(f)
    for tup,sents in wjs.items():
        tup = tuple(tup.split(', '))
        wj_nli[tup] = set(sents)


