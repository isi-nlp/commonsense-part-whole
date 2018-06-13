"""
    Start with google unigram counts and wordnet concreteness.
    Then, combine with either visual genome or PWKB relations to get part-whole relations
    Filter the parts to get rid of adjectives and stuff, and keep only frequent/concrete nouns
"""
import argparse
import copy
import csv
import operator
import re
import sys
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter, defaultdict

import spacy
from tqdm import tqdm

##########################################
# Begin adapted code, from Maxwell Forbes

ABSTRACT_SS_NAMES = ['abstraction.n.01',
                     'abstraction.n.02',
                     'abstraction.n.03',
                     'abstraction.n.04',
                     'abstractedness.n.01',
                     'abstraction.n.06']
ABSTRACT_SS = set([wn.synset(x) for x in ABSTRACT_SS_NAMES])
CONCRETE_SS = wn.synsets('physical_entity')[0]

def is_abstract(noun, all_senses=False):
    """
        Try to (heuristically) filter abstract nouns.

        Args:
            noun (str)
    """
    assert type(noun) == str, 'bad noun type: %r. noun: %s' % (type(noun), noun)
    noun_ss = wn.synsets(noun, pos=wn.NOUN)

    # if wordnet doesn't know about the noun, filter it out
    if len(noun_ss) == 0:
        return True

    # if *any* noun sense of the word is abstract, it is abstract (this may prove to be too restrictive)
    paths_good = True
    if all_senses:
        for n_ss in noun_ss:
            paths = n_ss.hypernym_paths()
            # all ps must pass
            for p in paths:
                if len(set(p).intersection(ABSTRACT_SS)) > 0:
                    paths_good = False
                    break
            if not paths_good:
                break
    else:
        #check only first (most common) sense
        paths = noun_ss[0].hypernym_paths()
        for p in paths:
            if len(set(p).intersection(ABSTRACT_SS)) > 0:
                paths_good = False
                break
    return not paths_good

def is_concrete(noun, all_senses=False):
    """
        Try to (heuristically) select concrete nouns.

        Args:
            noun (str)
    """
    assert type(noun) == str, 'bad noun type: %r. noun: %s' % (type(noun), noun)
    noun_ss = wn.synsets(noun, pos=wn.NOUN)

    # if wordnet doesn't know about the noun, filter it out
    if len(noun_ss) == 0:
        return False

    found_concrete = False
    if all_senses:
        # *forward select* those nouns where *at least one* sense is a physical entity (this may prove to be too permissive)
        for n_ss in noun_ss:
            paths = n_ss.hypernym_paths()
            for p in paths:
                if CONCRETE_SS in p:
                    found_concrete = True
                    break
            if found_concrete:
                break
    else:
        #check only first (most common) sense
        paths = noun_ss[0].hypernym_paths()
        for p in paths:
            if CONCRETE_SS in p:
                found_concrete = True
                break
    return found_concrete

# End adapted code
#######################################

def get_frequent_concrete_nouns(ngram_source, lem, min_count=10000):
    # load the candidates from google unigram noun counts
    print("frequency and concreteness filtering from google unigrams...")
    unigram_concrete = set()
    if ngram_source == 'full':
        nc_file = '/home/jamesm/commonsense/data/ngrams/unigrams/noun_counts.csv'
    elif ngram_source == 'fiction':
        nc_file = '/home/jamesm/commonsense/data/ngrams/unigrams_fic/noun_counts.csv'
    elif ngram_source == 'fiction-1950':
        nc_file = '/home/jamesm/commonsense/data/ngrams/unigrams_fic/noun_counts_1950.csv'
    print("reading from %s" % nc_file)
    with open(nc_file) as f:
        r = csv.reader(f)
        for row in tqdm(r):
            #baseline frequency filtering
            if int(row[1]) >= min_count:
                noun = row[0]
                #concreteness filtering
                if not is_abstract(noun) and is_concrete(noun):
                    unigram_concrete.add(noun)

    #lemmatize
    unigram_concrete_lem = set()
    for cand in unigram_concrete:
        unigram_concrete_lem.add(lem.lemmatize(cand))

    return unigram_concrete_lem

def get_part_whole_relations_vg(lem, min_count=1):
    # whole-ness filtering: compare with visual genome candidates
    fname = '../data/nouns/vg_has_nouns_tfidf_aggressive_stats.csv'
    data_inds = (0,1)
    print("reading whole nouns from visual genome...")
    whole2parts = defaultdict(set)
    whole_counts = {}; part_counts = {}; pw_counts = {}
    wholes_seen_unlem = set(); parts_seen_unlem = set(); pws_seen_unlem = set()
    with open(fname) as f:
        r = csv.reader(f)
        next(r)
        for row in tqdm(r):
            whole_unlem = row[data_inds[0]]
            part_unlem = row[data_inds[1]]
            whole = lem.lemmatize(whole_unlem)
            part = lem.lemmatize(part_unlem)
            whole2parts[whole].add(part)
            if whole in whole_counts and whole_unlem not in wholes_seen_unlem:
                #different lemma, add the values
                whole_counts[whole] += int(row[2])
            elif whole not in wholes_seen_unlem:
                whole_counts[whole] = int(row[2])
            if part in part_counts and part_unlem not in parts_seen_unlem:
                #different lemma, add the values
                part_counts[part] += int(row[3])
            elif part not in parts_seen_unlem:
                part_counts[part] = int(row[3])
            if (whole, part) in pw_counts and (whole_unlem, part_unlem) not in pws_seen_unlem:
                #different lemma combination, add the values
                pw_counts[(whole, part)] += int(row[4])
            elif (whole, part) not in pws_seen_unlem:
                pw_counts[(whole, part)] = int(row[4])
            wholes_seen_unlem.add(whole_unlem)
            parts_seen_unlem.add(part_unlem)
            pws_seen_unlem.add((whole_unlem, part_unlem))
    return whole2parts, (whole_counts, part_counts, pw_counts)

def get_part_whole_relations_cn(lem):

    #get locations, which are a big class we don't want that survive all the filters
    locations = set()
    with open('../data/conceptnet/conceptnet-isa-en.csv') as f:
        r = csv.reader(f, delimiter='\t')
        for row in r:
            if row[3].split('/')[3] in ['city', 'administrative_region', 'country', 'continent']:
                locations.add(re.sub('_', ' ', row[2].split('/')[3]))

    whole2parts_cn = defaultdict(set)
    print("reading whole nouns from conceptnet...")
    with open('../data/conceptnet/conceptnet-partof-en.csv') as f:                                                                                                                                    
        r = csv.reader(f, delimiter='\t')
        for i, row in tqdm(enumerate(r)):
            part = re.sub('_', ' ', row[2].split('/')[3])
            whole = re.sub('_', ' ', row[3].split('/')[3])
            #limit to single word concepts and not locations
            if len(whole.split()) == 1 and len(part.split()) == 1 and part not in locations and whole not in locations:
                whole2parts_cn[whole].add(part)
    return whole2parts_cn

def get_part_whole_relations_pwkb(lem):
    # whole-ness filtering: compare with pwkb candidates. Include only PWKB entries with score 1, others are mostly junk
    fname = '../data/pwkb/pwkb.txt'
    print("reading whole nouns from PWKB...")
    whole2parts = defaultdict(set)
    with open(fname) as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i,row in tqdm(enumerate(r)):
            if float(row[-1]) == 1.0:
                whole = lem.lemmatize(row[1])
                part = lem.lemmatize(row[4])
                whole2parts[whole].add(part)

    print("filtering PWKB relations...")
    #filter a whole's parts down to just the parts that distinguish it from its hypernyms
    #first, lookups
    no_ss = set()
    whole2ss = {}
    for whole in whole2parts.keys():
        if len(wn.synsets(whole)) > 0:
            whole2ss[whole] = wn.synsets(whole)[0]
        else:
            no_ss.add(whole)
    ss2wholes = defaultdict(set)
    for w,ss in whole2ss.items():
        ss2wholes[ss].add(w)
    wholes = set(whole2ss.keys())
    whole_ss = set(whole2ss.values())

    #keep only the "distinguishing" parts of a whole noun (so that 'actor' doesn't have all the body parts)
    whole2parts_disting = copy.deepcopy(whole2parts)
    for whole in tqdm(wholes):
         paths = whole2ss[whole].hypernym_paths()
         for p in paths:
             for i,hyper in enumerate(p):
                 #skip self-reference
                 if i == len(p) - 1:
                     continue
                 if hyper in whole_ss:
                     for hyper_whole in ss2wholes[hyper]:
                         whole2parts_disting[whole] = whole2parts_disting[whole] - whole2parts_disting[hyper_whole]

    #doc frequencies (to catch things like 'chromosome' that persist)
    dfs = Counter()
    for whole, parts in whole2parts_disting.items():
         for part in parts:
             dfs[part] += 1
    #optional: filter out those with high df (from observation this is mostly things like 'chromosome' that don't get completely filtered out by the distinguishing step
    whole2parts_disting = defaultdict(set, {whole: set([part for part in parts if dfs[part] < 1000]) for whole, parts in whole2parts_disting.items()})
    return whole2parts_disting

"""
def filter_parts(whole2parts):
    nlp = spacy.load('en_core_web_sm', parser=False, tagger=True, entity=False)

    # build set of parts
    parts = set()
    for noun, partset in whole2parts.items():
        parts.update(set([*partset]))

    #filter out adjectives and other stuff that makes parts messy (maybe applicable to VG only)
    #iterate over the set of parts and make a lookup so we only have to do the pos tagging once
    part2filtered = {}
    for part in tqdm(parts):
         doc = nlp(part)
         noun_part = None
         #select the (UD tagset) noun (if multiple, heuristically choose the last one)
         for tok in doc:
             if tok.pos_ == 'NOUN':
                 noun_part = tok
         if noun_part:
             part2filtered[part] = noun_part.text

    #ignore parts that don't have nouns in them
    whole2parts_filt = {noun: set([part2filtered[part] for part in partset if part in part2filtered]) for noun, partset in whole2parts.items()}

    #re-build set of parts
    parts_filt = set()
    for noun, partset in whole2parts_filt.items():
        parts_filt.update(set([*partset]))

    #filter part nouns by unigram frequency and concreteness
    candidate_parts = parts_filt.intersection(unigram_concrete)

    #ignore parts not in candidate parts
    whole2parts_filt = {noun: set([part2filtered[part] for part in partset if part in part2filtered and part in candidate_parts]) for noun, partset in whole2parts.items()}
    return whole2parts_filt
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pw_source', choices=['vg', 'pwkb', 'cn', 'all', 'no-vg', 'no-pwkb'], help='Data source for part-whole relations')
    parser.add_argument('ngram_source', choices=['full', 'fiction', 'fiction-1950'], help='Data source for google unigram frequencies')
    parser.add_argument('--concreteness', type=float, required=False, default=3.5, help="cutoff lower bound for brysbaert's concreteness judgment (default: 3.5)")
    parser.add_argument('--noun-freq', type=int, required=False, default=10000, dest='noun_freq', help="minimum unigram noun frequency from google n-grams")
    parser.add_argument('--vg-freq', type=int, required=False, default=1, dest='vg_freq', help="minimum visual genome image frequency for parts (unimplemented)")
    parser.add_argument('--union-concrete', dest='union_concrete', const=True, required=False, action="store_const", help="flag to union brysbaert with google ngrams rather than intersect")
    args = parser.parse_args()

    # First, get frequent, concrete nouns using google n-grams and wordnet
    lem = WordNetLemmatizer()
    unigram_concrete = get_frequent_concrete_nouns(args.ngram_source, lem, args.noun_freq)
    #vg_sanity = get_frequent_concrete_nouns(args.ngram_source, lem, 1000)

    # Augment (or filter) with most concrete of brysbaert's concrete lemmas from turkers
    concs = {}
    conc_sds = {}                                                 
    with open('/home/jamesm/commonsense/data/brysbaert/Concreteness_ratings_Brysbaert_et_al_BRM.txt') as f:                                                                                           
        r = csv.reader(f, delimiter='\t')
        next(r)                                                                                                                                                                                       
        for row in r:             
            if row[-1] == "Noun":
                concs[row[0]] = float(row[2])
                conc_sds[row[0]] = float(row[3])
    br_nouns = set([lem.lemmatize(noun) for noun in concs.keys() if concs[noun] - conc_sds[noun] > args.concreteness])
    if args.union_concrete:
        unigram_concrete = unigram_concrete.union(br_nouns)
    else:
        unigram_concrete = unigram_concrete.intersection(br_nouns)
    print("NUM FREQUENT CONCRETE UNIGRAM LEMMAS: %d" % len(unigram_concrete))

    # Get part-whole relations from visual genome, part-whole KB, or both
    if args.pw_source == 'vg':
        whole2parts, vg_stats = get_part_whole_relations_vg(lem, args.vg_freq)
    elif args.pw_source == 'pwkb':
        whole2parts = get_part_whole_relations_pwkb(lem)
    elif args.pw_source == 'cn':
        whole2parts = get_part_whole_relations_cn(lem)
    elif args.pw_source == 'no-pwkb':
        whole2parts_vg, vg_stats = get_part_whole_relations_vg(lem, args.vg_freq)
        whole2parts_cn = get_part_whole_relations_cn(lem)

        # Create set of concrete *whole* nouns
        #unigram_concrete_whole = unigram_concrete.intersection(set(whole2parts_cn.keys()))
        ## Create set of concrete *part* nouns
        #parts = set()
        #for noun, partset in whole2parts_cn.items():
        #    parts.update(set([*partset]))
        #unigram_concrete_part = unigram_concrete.intersection(parts)

        ## filter down both wholes and parts in ConceptNet lookup
        #whole2parts_cn = defaultdict(set, {noun: set([part for part in partset if part in unigram_concrete_part and part != noun])
        #                                   for noun,partset in whole2parts_cn.items() if noun in unigram_concrete_whole})
        #whole2parts_vg = defaultdict(set, {noun: set([part for part in partset if part in vg_sanity and part != noun])
        #                                   for noun,partset in whole2parts_vg.items() if noun in vg_sanity})

        #combine by union
        whole2parts = defaultdict(set, {whole: whole2parts_vg[whole].union(whole2parts_cn[whole]) \
                                        for whole in set(whole2parts_vg.keys()).union(set(whole2parts_cn.keys()))})
        wp2source = defaultdict(set)
        for whole, parts in whole2parts.items():
            for part in parts:
                if part in whole2parts_vg[whole]:
                    wp2source[(whole, part)].add('vg')
                if part in whole2parts_cn[whole]:
                    wp2source[(whole, part)].add('cn')
    elif args.pw_source == 'no-vg':
        whole2parts_pwkb = get_part_whole_relations_pwkb(lem)
        whole2parts_cn = get_part_whole_relations_cn(lem)

        # Create set of concrete *whole* nouns
        #unigram_concrete_whole = unigram_concrete.intersection(set(whole2parts_cn.keys()))
        ## Create set of concrete *part* nouns
        #parts = set()
        #for noun, partset in whole2parts_cn.items():
        #    parts.update(set([*partset]))
        #unigram_concrete_part = unigram_concrete.intersection(parts)

        ## filter down both wholes and parts in ConceptNet lookup
        #whole2parts_cn = defaultdict(set, {noun: set([part for part in partset if part in unigram_concrete_part and part != noun])
        #                                   for noun,partset in whole2parts_cn.items() if noun in unigram_concrete_whole})
        #whole2parts_vg = defaultdict(set, {noun: set([part for part in partset if part in vg_sanity and part != noun])
        #                                   for noun,partset in whole2parts_vg.items() if noun in vg_sanity})

        #combine by union
        whole2parts = defaultdict(set, {whole: whole2parts_pwkb[whole].union(whole2parts_cn[whole]) \
                                        for whole in set(whole2parts_pwkb.keys()).union(set(whole2parts_cn.keys()))})
        wp2source = defaultdict(set)
        for whole, parts in whole2parts.items():
            for part in parts:
                if part in whole2parts_pwkb[whole]:
                    wp2source[(whole, part)].add('pwkb')
                if part in whole2parts_cn[whole]:
                    wp2source[(whole, part)].add('cn')


    elif args.pw_source == 'all':
        whole2parts_vg, vg_stats = get_part_whole_relations_vg(lem, args.vg_freq)
        whole2parts_pwkb = get_part_whole_relations_pwkb(lem)
        whole2parts_cn = get_part_whole_relations_cn(lem)
        #combine by union
        whole2parts = defaultdict(set, {whole: whole2parts_vg[whole].union(whole2parts_pwkb[whole]).union(whole2parts_cn[whole]) \
                                        for whole in set(whole2parts_vg.keys()).union(set(whole2parts_pwkb.keys())).union(set(whole2parts_cn.keys()))})
        wp2source = defaultdict(set)
        for whole, parts in whole2parts.items():
            for part in parts:
                if part in whole2parts_vg[whole]:
                    wp2source[(whole, part)].add('vg')
                if part in whole2parts_pwkb[whole]:
                    wp2source[(whole, part)].add('pwkb')
                if part in whole2parts_cn[whole]:
                    wp2source[(whole, part)].add('cn')

    # Create set of concrete *whole* nouns
    unigram_concrete_whole = unigram_concrete.intersection(set(whole2parts.keys()))
    # Create set of concrete *part* nouns
    parts = set()
    for noun, partset in whole2parts.items():
        parts.update(set([*partset]))
    unigram_concrete_part = unigram_concrete.intersection(parts)

    # filter down both wholes and parts in lookup
    whole2parts = {noun: set([part for part in partset if part in unigram_concrete_part and part != noun]) for noun,partset in whole2parts.items() if noun in unigram_concrete_whole}

    #write!
    print("TOTAL WHOLES: %d" % len(set(whole2parts.keys())))
    print("TOTAL PARTS: %d" % sum([len(parts) for parts in whole2parts.values()]))
    suffix = "_union" if args.union_concrete else ""
    with open('/home/jamesm/commonsense/data/nouns/part_whole_candidates_%s_%s_%d%s_stats.csv' % (args.ngram_source, args.pw_source, args.noun_freq, suffix), 'w') as of:
        w = csv.writer(of)
        w.writerow(['whole', 'part', 'sources', 'vg_whole', 'vg_part', 'vg_part_whole'])
        for whole, parts in whole2parts.items():
            for part in parts:
                source = wp2source[(whole, part)]
                if whole in vg_stats[0] and part in vg_stats[1] and (whole, part) in vg_stats[2]:
                    w.writerow([whole, part, ';'.join(source), vg_stats[0][whole], vg_stats[1][part], vg_stats[2][(whole, part)]])
                else:
                    w.writerow([whole, part, ';'.join(source)])
