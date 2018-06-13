"""
    Extract candidate adjectives from google dependency n-grams, visual genome attributes. Ignore jonathan's lists
"""
import csv, json
from collections import Counter, defaultdict, OrderedDict
import operator

print("reading adjective lists")
jjs_to_exclude = set()
for typ in ['descriptive', 'attributive']:
    with open('/home/jamesm/commonsense/data/adjectives/non-%s-jj.txt' % typ) as f:
        for line in f:
            jjs_to_exclude.add(line.strip())

with open('/home/jamesm/commonsense/data/adjectives/non-descript-jj-james.txt') as f:
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
with open('../data/nouns/sample_50.tsv') as f:
    r = csv.reader(f, delimiter='\t')
    for row in r:
        whole2parts[row[0]].add(row[1])

#intersect b/c some wholes might not have any adjectives
wholes = set(whole2parts.keys()).intersection(set(noun2jjs.keys()))

#take top 5 adjectives for the whole
with open('../data/adjectives/sample_50_pws.csv', 'w') as of:
    w = csv.writer(of)
    for whole in wholes:
        #take top five adjectives from n-grams
        jjs = list(noun2jjs[whole])[:5]
        #optionally take top five adjectives from VG as well
        if whole in noun2jjs_vg:
            jjs.extend(list(noun2jjs_vg[whole])[:5])
        for jj in jjs:
            for part in whole2parts[whole]:
                #optionally filter to adjectives that have been applied to the part (in n-grams) as well
                if part in noun2jjs and jj in noun2jjs[part]:
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
