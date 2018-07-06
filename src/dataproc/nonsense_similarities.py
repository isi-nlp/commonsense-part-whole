# coding: utf-8
words = []
for line in open('non_vis_sample.csv'):
    words.extend(line.strip().split(',')[:3])
    
words = sorted(set(words))
len(words)
type(words)
words[:5]
embeds = {}
from tqdm import tqdm
with open('../../lib/glove/glove.840B.300d.txt') as f:
    for line in tqdm(f):
        row = line.split()
        if row[0] in vocab:
            embeds[row[0]] = [float(r) for r in row[1:]]
            seen.add(row[0])
            
with open('../../lib/glove/glove.840B.300d.txt') as f:
    for line in tqdm(f):
        row = line.split()
        if row[0] in words:
            embeds[row[0]] = [float(r) for r in row[1:]]
            seen.add(row[0])
            
seen = set()
with open('../../lib/glove/glove.840B.300d.txt') as f:
    for line in tqdm(f):
        row = line.split()
        if row[0] in words:
            embeds[row[0]] = [float(r) for r in row[1:]]
            seen.add(row[0])
            
seen = set()
embeds = {}
#for fun
printed_3 = False
printed_1 = False
with open('../../lib/glove/glove.840B.300d.txt') as f:
    for line in tqdm(f):
        row = line.split()
        if row[0] in vocab:
            embeds[row[0]] = [float(r) for r in row[1:]]
            seen.add(row[0])

        unseen = vocab.difference(seen)
        if len(unseen) == 0:
            break
        elif len(unseen) == 3 and not printed_3:
            print("still haven't seen these three")
            print(unseen)
            print()
            printed_3 = True
        elif len(unseen) == 1 and not printed_1:
            print("still haven't seen this one!")
            print(unseen)
            print()
            printed_1 = True
words = set(words)
seen = set()
embeds = {}
#for fun
printed_3 = False
printed_1 = False
with open('../../lib/glove/glove.840B.300d.txt') as f:
    for line in tqdm(f):
        row = line.split()
        if row[0] in words:
            embeds[row[0]] = [float(r) for r in row[1:]]
            seen.add(row[0])

        unseen = words.difference(seen)
        if len(unseen) == 0:
            break
        elif len(unseen) == 3 and not printed_3:
            print("still haven't seen these three")
            print(unseen)
            print()
            printed_3 = True
        elif len(unseen) == 1 and not printed_1:
            print("still haven't seen this one!")
            print(unseen)
            print()
            printed_1 = True
get_ipython().run_line_magic('ls', '')
embeds['hat']
embeds['rampart']
type(embeds['rampart'])
import numpy as np
np.dot(embeds['rampart'], embeds['battlement'])
import scipy.spatial.distance.cosine as cossim
import scipy.spatial
import scipy.spatial as spatial
spatial.distance
spatial.distance.cosine
from scipy.spatial.distance import cosine as cossim
cossim(embeds['rampart'], embeds['battlement'])
cossim(embeds['rampart'], embeds['apron'])
sims = {}
for word, vec in embeds.items():
    sims[word] = cossim(embeds['rampart'], vec)
    
sims
words = set(words)
seen = set()
embeds = {}
#for fun
printed_3 = False
printed_1 = False
with open('../../lib/glove/glove.840B.300d.txt') as f:
    for line in tqdm(f):
        row = line.split()
        if row[0] in words:
            embeds[row[0]] = np.array([float(r) for r in row[1:]])
            seen.add(row[0])

        unseen = words.difference(seen)
        if len(unseen) == 0:
            break
        elif len(unseen) == 3 and not printed_3:
            print("still haven't seen these three")
            print(unseen)
            print()
            printed_3 = True
        elif len(unseen) == 1 and not printed_1:
            print("still haven't seen this one!")
            print(unseen)
            print()
            printed_1 = True
sims = {}
for word, vec in embeds.items():
    sims[word] = cossim(embeds['rampart'], vec)
    
sims
import operator
sorted(sims.items(), key=operator.itemgetter(1), reverse=True)[:10]
def most_similar_for_word(word1):
    sims = {}
    for word2, vec in embeds.items():
        sims[word2] = cossim(embeds[word1], vec)
    print(sorted(sims.items(), key=operator.itemgetter(1), reverse=True)[:10])
    
list(embeds.keys())
list(embeds.keys())[:10]
most_similar_for_word('back')
most_similar_for_word('car')
most_similar_for_word('public')
def most_similar_for_word(word1):
    sims = {}
    for word2, vec in embeds.items():
        sims[word2] = -1*(cossim(embeds[word1], vec))+1
    print(sorted(sims.items(), key=operator.itemgetter(1), reverse=True)[:10])
    
most_similar_for_word('public')
most_similar_for_word('back')
def most_similar_for_word(word1):
    sims = {}
    for word2, vec in embeds.items():
        if word1 != word2:
            sims[word2] = -1*(cossim(embeds[word1], vec))+1
    print(sorted(sims.items(), key=operator.itemgetter(1), reverse=True)[:10])
    
most_similar_for_word('back')
def similarity(word1, word2):
    return -1*(cossim(embeds[word1], embeds[word2]))+1
import os; os.getcwd()
for line in open('non_vis_sample.csv'):
    print(line)
    vals = line.strip().split(',')
    print("part-whole similarity: {:.4f}".format(similarity(vals[0], vals[1])))
    
nonsense_sims = []
sense_sims = []
for line in open('non_vis_sample.csv'):
    print(line)
    vals = line.strip().split(',')
    sim = similarity(vals[0], vals[1])
    print("part-whole similarity: {:.4f}".format(sim))
    if 'nonsense' in line:
        nonsense_sims.append(sim)
    else:
        sense_sims.append(sim)
    
np.mean(nonsense_sims)
np.mean(sense_sims)
np.mean(nonsense_sims), np.std(nonsense_sims)
np.mean(sense_sims), np.std(sense_sims)
len(nonsense_sims > .4)
nonsense_sims > .4
np.array(nonsense_sims) > .4
sum(np.array(nonsense_sims) > .4)
len(nonsense_sims)
sum(np.array(sense_sims) < .4)
len(sense_sims)
nonsense_sims = []
sense_sims = []
for line in open('non_vis_sample.csv'):
    print(line)
    vals = line.strip().split(',')
    sim = similarity(vals[0], vals[1])
    print("part-whole similarity: {:.4f}".format(sim))
    if 'pw-nonsense' in line:
        nonsense_sims.append(sim)
    else:
        sense_sims.append(sim)
    
np.mean(sense_sims), np.std(sense_sims)
np.mean(nonsense_sims), np.std(nonsense_sims)
