"""
    Extract candidate adjectives from google dependency n-grams, visual genome attributes. Ignore jonathan's lists
"""
import argparse
import csv, json
from collections import Counter, defaultdict, OrderedDict
import operator

from nltk.corpus import wordnet as wn
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('pw_dataset', type=str, help="filename of part-whole candidates")
parser.add_argument('out_file', type=str, help="filename to write to")
parser.add_argument('--non-visual', const=True, action="store_const", required=False, dest='non_visual', help="flag to get non-visual adjectives")
args = parser.parse_args()

print("reading adjective lists")
jjs_to_exclude = set()
for typ in ['descriptive', 'attributive']:
    with open('/home/jamesm/commonsense-part-whole/data/adjectives/non-%s-jj.txt' % typ) as f:
        for line in f:
            jjs_to_exclude.add(line.strip())

with open('/home/jamesm/commonsense-part-whole/data/adjectives/non-descript-jj-james.txt') as f:
    for line in f:
        jjs_to_exclude.add(line.strip())


print("reading noun-adjective counts from google dependency n-grams...")
with open('../data/ngrams/nn-jj-amod/nn.jj.amod.all') as f:
    r = csv.reader(f, delimiter=' ')
    noun2jjs = {}
    for row in r:
        noun = row[0]
        jjs = OrderedDict()
        for i in range(1,len(row),2):
            if row[i] not in jjs_to_exclude:
                jjs[row[i]] = int(row[i+1])
        noun2jjs[noun] = jjs

print("reading noun-adjective counts from visual genome...")
attrs = json.load(open('../data/visualgenome/attributes.json'))
noun2jjs_vg = defaultdict(Counter)
for img in attrs:
    for attr in img['attributes']:
        if 'attributes' in attr:
            for jj in attr['attributes']:
                noun2jjs_vg[attr['names'][0]][jj] += 1
 
print("reading part-whole candidates")
whole2parts = defaultdict(set)
with open('../data/nouns/%s' % args.pw_dataset) as f:
    delim = ',' if args.non_visual else '\t'
    r = csv.reader(f, delimiter=delim)
    #header
    next(r)
    for row in r:
        whole2parts[row[0]].add(row[1])

wholes = set(whole2parts.keys())

if args.non_visual:
    triples = set([tuple(row) for row in csv.reader(open('../data/adjectives/vg_only_mturk_candidates_mwe.csv'))])

#take top 5 adjectives for the whole
with open('../data/adjectives/%s' % args.out_file, 'w') as of:
    w = csv.writer(of)
    for whole in tqdm(wholes):

        jjs = []
        #take top five adjectives from n-grams
        if whole in noun2jjs:
            jjs.extend(list(noun2jjs[whole])[:5])
            whole_for_adj = whole
        #for multi-word nouns, try taking adjectives for the second noun
        elif ' ' in whole and whole.split(' ')[1] in noun2jjs_vg:
            jjs.extend(list(noun2jjs[whole.split(' ')[1]])[:5])
            whole_for_adj = whole.split(' ')[1]

        #optionally take top five adjectives from VG as well
        if whole in noun2jjs_vg:
            jjs.extend(list(noun2jjs_vg[whole])[:5])
        elif ' ' in whole and whole.split(' ')[1] in noun2jjs_vg:
            jjs.extend(list(noun2jjs_vg[whole.split(' ')[1]])[:5])

        if len(jjs) == 0:
            print("whole had no jj's found: %s" % whole)
            continue

        for jj in jjs:
            #filter jjs where the jj-whole forms a common expression (in wordnet) (e.g. dutch oven, sick bed)
            if len(wn.synsets('_'.join([jj, whole_for_adj]))) == 0 and len(wn.synsets(''.join([jj, whole_for_adj]))) == 0:
                for part in whole2parts[whole]:
                    if (not args.non_visual) or ((whole, part, jj) not in triples):
                        #optionally filter to adjectives that have been applied to the part (in n-grams) as well
                        if part in noun2jjs:
                            if jj in noun2jjs[part]:
                                if part != whole and '.' not in whole and '.' not in part:
                                    w.writerow([whole, part, jj])
                        elif ' ' in part and part.split(' ')[1] in noun2jjs:
                            if jj in noun2jjs[part.split(' ')[1]]:
                                if part != whole and '.' not in whole and '.' not in part:
                                    w.writerow([whole, part, jj])

#put jjs in vg_imgs directories
#for whole in wholes:
#    #take top five adjectives from n-grams
#    jjs = list(noun2jjs[whole])[:5]
#    #optionally take top five adjectives from VG as well
#    if whole in noun2jjs_vg:
#        jjs.extend(list(noun2jjs_vg[whole])[:5])
#    for jj in jjs:
#        for part in whole2parts[whole]:
#            #optionally filter to adjectives that have been applied to the part as well
#            if part in noun2jjs and jj in noun2jjs[part]:
#                pw = '_'.join([whole, part])
#                with open('../data/nouns/vg_imgs/%s/jjs.csv' % pw, 'a') as of:
#                    of.write(','.join([whole, part, jj]) + '\n')
