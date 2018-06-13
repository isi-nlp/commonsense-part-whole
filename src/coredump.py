# coding: utf-8
from noun_candidates import *
unigram_concrete = get_frequent_concrete_nouns()
noun2parts = get_part_whole_relations_pwkb()
unigram_concrete_whole = unigram_concrete.intersection(set(noun2parts.keys()))
len(unigram_concrete_whole)
lem = WordNetLemmatizer()
unigram_concrete_whole_lem = set()
for cand in unigram_concrete_whole:
    unigram_concrete_whole_lem.add(lem.lemmatize(cand))
nlp = spacy.load('en_core_web_sm', parser=False, tagger=True, entity=False)
parts = set()
for noun, partset in noun2parts.items():
    parts.update(set([*partset]))
len(unigram_concrete_whole_lem)
len(parts)
sorted(list(parts))[:10]
sorted(list(parts))[100:150]
sorted(list(parts))[1000:1050]
sorted(list(parts))[-50:]
len(parts)
part2wholes = defaultdict(set)
for noun, partset in noun2parts.items():
    for part in partset:
        part2wholes[part].add(noun)
        
len(part2wholes)
part2wholes['yellow-breasted bunting']
part2wholes['yoke']
sorted(list(parts))[-150:-100]
part2wholes['wool']
part2wholes['worm']
part2wholes['womb']
sorted(list(parts))[-150:-100]
part2wholes['wombat']
part2wholes['wolf']
part2wholes['worker']
is_abstract('labor')
'labor' in unigram_concrete
'labor' in unigram_concrete_whole
noun2parts = {noun:partset for noun,partset in noun2parts.items() if noun in unigram_concrete_whole_lem}
len(noun2parts)
parts = set()
for noun, partset in noun2parts.items():
    parts.update(set([*partset]))
len(parts)
sorted(list(parts))
sorted(list(noun2parts.keys()))
noun2parts['bodybuilder']
sorted(list(noun2parts.keys()))
noun2parts['blunderbuss']
'cartridge ejector' in parts
'structural member' in parts
noun2parts = {noun: set([part for part in partset if part in unigram_concrete_whole_lem]) for noun,partset in noun2parts.items() if noun in unigram_concrete_whole_lem}
len(noun2parts)
parts = set()
for noun, partset in noun2parts.items():
    parts.update(set([*partset]))
len(parts)
noun2parts['bodybuilder']
sorted(list(parts))
part2wholes = defaultdict(set)
for noun, partset in noun2parts.items():
    for part in partset:
        part2wholes[part].add(noun)
        
part2wholes['soundtrack']
part2wholes['spearhead']
part2wholes['snowflake']
part2wholes['speedometer']
noun2parts['ambulance']
import random
random.choice(noun2parts.items())
random.choice(list(noun2parts.items()))
random.choice(list(noun2parts.items()))
random.choice(list(noun2parts.items()))
len(unigram_concrete)
'misogynist' in unigram_concrete
unigram_concrete = get_frequent_concrete_nouns(min_count=1e5)
unigram_concrete_whole = unigram_concrete.intersection(set(noun2parts.keys()))
len(unigram_concrete_whole)
'misogynist' in unigram_concrete_whole
'trochanter' in unigram_concrete_whole
'trochanter' in unigram_concrete
len(unigram_concrete)
'tibialis' in unigram_concrete
'trachea' in unigram_concrete
'zygoma' in unigram_concrete
lem = WordNetLemmatizer()
unigram_concrete_whole_lem = set()
for cand in unigram_concrete_whole:
    unigram_concrete_whole_lem.add(lem.lemmatize(cand))
len(unigram_concrete_whole_lem)
noun2parts = {noun: set([part for part in partset if part in unigram_concrete_whole_lem]) for noun,partset in noun2parts.items() if noun in unigram_concrete_whole_lem}
len(noun2parts)
parts = set()
for noun, partset in noun2parts.items():
    parts.update(set([*partset]))
len(parts)
sorted(list(parts))[:50]
len(unigram_concrete)
unigram_concrete = get_frequent_concrete_nouns(min_count=5e5)
len(unigram_concrete)
unigram_concrete_whole = unigram_concrete.intersection(set(noun2parts.keys()))
lem = WordNetLemmatizer()
unigram_concrete_whole_lem = set()
for cand in unigram_concrete_whole:
    unigram_concrete_whole_lem.add(lem.lemmatize(cand))
noun2parts = {noun: set([part for part in partset if part in unigram_concrete_whole_lem]) for noun,partset in noun2parts.items() if noun in unigram_concrete_whole_lem}
len(noun2parts)
parts = set()
for noun, partset in noun2parts.items():
    parts.update(set([*partset]))
len(parts)
sorted(list(parts))[:50]
sorted(list(parts))[-50:]
part2wholes = defaultdict(set)
for noun, partset in noun2parts.items():
    for part in partset:
        part2wholes[part].add(noun)
        
part2wholes['yolk']
noun2parts['brill']
wn
wn.synsets('brill')
wn.synsets('brill')[0].hypernym_paths()
noun2parts['brill'] - noun2parts['fish']
part2wholes['yolk']
noun2parts['sleeper'] - noun2parts['fish']
part2wholes['yolk']
noun2parts['salmon'] - noun2parts['fish']
noun2parts['herring'] - noun2parts['fish']
noun2parts['eel'] - noun2parts['fish']
noun2parts['drum'] - noun2parts['fish']
wn.synsets('eel')[0]
wn.synsets('eel')[0].hypernym_paths()
noun2parts['food']
wholes = set(noun2parts.keys())
whole_ss = set([wn.synsets(whole)[0] for whole in noun2parts.keys()])
len(whole_ss)
len(noun2parts)
set(noun2parts.keys()) - whole_ss
wn.synsets('federalist')
wn.synsets('guy')
wn.synsets('dude')
wn.synsets('man')
len(whole_ss)
whole_ss
for ss in whole_ss:
    paths = ss.hypernym_paths():
whole_ss = set([wn.synsets(whole)[0] for whole in noun2parts.keys()])
whole2ss = {whole:wn.synsets(whole)[0] for whole in noun2parts.keys()]}
whole2ss = {whole:wn.synsets(whole)[0] for whole in noun2parts.keys()}
ss2whole = {ss:w for w,ss in whole2ss.items()}
whole_ss = set(whole2ss.values())
for ss in whole_ss:
    paths = ss.hypernym_paths():
#keep only 'distinguishing' parts
for ss in whole_ss:
    paths = ss.hypernym_paths()
    for p in paths:
        for hyper in p:
            if hyper in whole_ss:
                noun2parts[ss2whole[ss]] = noun2parts[ss2whole[ss]] - noun2parts[ss2whole[hyper]]
                
noun2parts['brill']
noun2parts['fish']
noun2parts
noun2parts = get_part_whole_relations_pwkb()
noun2parts = {noun: set([part for part in partset if part in unigram_concrete_whole_lem]) for noun,partset in noun2parts.items() if noun in unigram_concrete_whole_lem}
len(noun2parts)
noun2parts['fish']
noun2parts['animal']
noun2parts['fish'] - noun2parts['animal']
wn.synsets('fish')[0]
wn.synsets('fish')[0].hypernym_paths()
noun2parts['organism']
noun2parts['whole']
noun2parts['object']
#keep only 'distinguishing' parts
for ss in whole_ss:
    break
    paths = ss.hypernym_paths()
    for p in paths:
        for hyper in p:
            if hyper in whole_ss:
                noun2parts[ss2whole[ss]] = noun2parts[ss2whole[ss]] - noun2parts[ss2whole[hyper]]
                
list(whole_ss)[0]
paths = list(whole_ss)[0].hypernym_paths()
paths
p = paths[0]
#keep only 'distinguishing' parts
for ss in whole_ss:
    paths = ss.hypernym_paths()
    for p in paths:
        for i,hyper in enumerate(p):
            #skip self-reference
            if i == 0:
                continue
            if hyper in whole_ss:
                noun2parts[ss2whole[ss]] = noun2parts[ss2whole[ss]] - noun2parts[ss2whole[hyper]]
noun2parts['learner']
noun2parts
noun2parts = get_part_whole_relations_pwkb()
noun2parts = {noun: set([part for part in partset if part in unigram_concrete_whole_lem]) for noun,partset in noun2parts.items() if noun in unigram_concrete_whole_lem}
len(noun2parts)
#keep only 'distinguishing' parts
for ss in whole_ss:
    break
    paths = ss.hypernym_paths()
    for p in paths:
        for i,hyper in enumerate(p):
            #skip self-reference
            if i == 0:
                continue
            if hyper in whole_ss:
                noun2parts[ss2whole[ss]] = noun2parts[ss2whole[ss]] - noun2parts[ss2whole[hyper]]
ss = list(whole_ss)[0]
ss
paths = ss.hypernym_paths()
p = paths[0]
for i,hyper in enumerate(p):
    print(i,hyper)
    
p
#keep only 'distinguishing' parts
for ss in whole_ss:
    paths = ss.hypernym_paths()
    for p in paths:
        for i,hyper in enumerate(p):
            #skip self-reference
            if i == len(p) - 1:
                continue
            if hyper in whole_ss:
                noun2parts[ss2whole[ss]] = noun2parts[ss2whole[ss]] - noun2parts[ss2whole[hyper]]
noun2parts
wn.synsets('hotel')
wn.synsets('hotel')[0].hypernym_paths()
noun2parts['building']
noun2parts['hotel']
noun2parts['ambulance']
noun2parts['motorcycle']
noun2parts['airplane']
noun2parts
noun2parts['proctor']
wn.synsets('proctor')
wn.synsets('proctor')[0].hypernym_paths()
noun2parts['person']
noun2parts['proctor']
wn.synsets('proctor')[0].hypernym_paths()
noun2parts['person']
whole2ss['person']
#keep only 'distinguishing' parts
for ss in whole_ss:
    break
    paths = ss.hypernym_paths()
    for p in paths:
        for i,hyper in enumerate(p):
            #skip self-reference
            if i == len(p) - 1:
                continue
            if hyper in whole_ss:
                noun2parts[ss2whole[ss]] = noun2parts[ss2whole[ss]] - noun2parts[ss2whole[hyper]]
paths = wn.synsets('proctor')[0].hypernym_paths()
p = paths[0]
p
ss = wn.synsets('proctor')[0]
ss
ss2whole[ss]
noun2parts['monitor']
whole2ss['monitor']
noun2parts['person']
noun2parts['monitor']
np.mean([len(parts) for noun, parts in noun2parts.items()])
import numpy as np
np.mean([len(parts) for noun, parts in noun2parts.items()])
np.median([len(parts) for noun, parts in noun2parts.items()])
for noun, parts in noun2parts.items():
    print(noun, parts[:5])
    
for noun, parts in noun2parts.items():
    print(noun, sorted(list(parts))[:5])
    
noun2parts['bologna']
wn.synsets('bologna']
wn.synsets('bologna')
wn.synsets('bologna')[0].hypernym_paths()
noun2parts = get_part_whole_relations_pwkb()
noun2parts = {noun: set([part for part in partset if part in unigram_concrete_whole_lem]) for noun,partset in noun2parts.items() if noun in unigram_concrete_whole_lem}
noun2parts_disting = defaultdict(set)
import copy
noun2parts_disting = copy.deepcopy(noun2parts)
#keep only 'distinguishing' parts
for ss in whole_ss:
    break
    paths = ss.hypernym_paths()
    for p in paths:
        for i,hyper in enumerate(p):
            #skip self-reference
            if i == len(p) - 1:
                continue
            if hyper in whole_ss:
                noun2parts_disting[ss2whole[ss]] = noun2parts_disting[ss2whole[ss]] - noun2parts_disting[ss2whole[hyper]]
for noun, parts in noun2parts.items():
    print(noun, sorted(list(parts))[:5])
    
len([n for n, p in noun2parts.items() if len(p) > 0])
len(noun2parts)
for noun, parts in noun2parts.items():
    if len(parts) > 0:
        print(noun, sorted(list(parts))[:5])
    
for noun, parts in noun2parts.items():
    if len(parts) > 3:
        print(noun, sorted(list(parts))[:5])
    
i = 0
for noun, parts in noun2parts.items():
    if len(parts) > 3:
        print(i, noun, sorted(list(parts))[:5])
        i += 1
        
noun2parts['mustard']
i = 0
for noun, parts in noun2parts.items():
    if len(parts) > 7:
        print(i, noun, sorted(list(parts))[:5])
        i += 1
        
i = 0
for noun, parts in noun2parts.items():
    if len(parts) > 7:
        print(i, noun, sorted(list(parts)))
        i += 1
        
i = 0
for noun, parts in noun2parts.items():
    if len(parts) > 7:
        print(i, noun, sorted(list(parts)[:7]))
        i += 1
        
wn.synsets('pea')[0].hypernym_paths()
noun2parts['vegetable']
i = 0
for noun, parts in noun2parts_disting.items():
    if len(parts) > 7:
        print(i, noun, sorted(list(parts)[:7]))
        i += 1
        
noun2parts_disting
#keep only 'distinguishing' parts
for ss in whole_ss:
    break
    paths = ss.hypernym_paths()
    for p in paths:
        for i,hyper in enumerate(p):
            #skip self-reference
            if i == len(p) - 1:
                continue
            if hyper in whole_ss:
                noun2parts_disting[ss2whole[ss]] = noun2parts_disting[ss2whole[ss]] - noun2parts_disting[ss2whole[hyper]]
noun2parts_disting
noun2parts_disting['hotel'] = {'haha'}
noun2parts_disting['hotel']
len(whole_ss)
whole2ss['pea']
#keep only 'distinguishing' parts
for ss in whole_ss:
    paths = ss.hypernym_paths()
    for p in paths:
        for i,hyper in enumerate(p):
            #skip self-reference
            if i == len(p) - 1:
                continue
            if hyper in whole_ss:
                noun2parts_disting[ss2whole[ss]] = noun2parts_disting[ss2whole[ss]] - noun2parts_disting[ss2whole[hyper]]
noun2parts_disting
whole2ss['millionaire']
ss = whole2ss['millionaire']
ss2whole['millionaire']
ss2whole[ss]
wn.synsets('millionaire')[0]
wn.synsets('millionaire')[0].hypernym_paths()
noun2parts['person']
noun2parts['millionaire']
noun2parts_disting['millionaire']
noun2parts_disting = copy.deepcopy(noun2parts)
#keep only 'distinguishing' parts
for ss in whole_ss:
    paths = ss.hypernym_paths()
    for p in paths:
        for i,hyper in enumerate(p):
            #skip self-reference
            if i == len(p) - 1:
                continue
            if hyper in whole_ss:
                print("removing %s parts from %s" % (ss2whole[hyper], ss2whole[ss]))
                noun2parts_disting[ss2whole[ss]] = noun2parts_disting[ss2whole[ss]] - noun2parts_disting[ss2whole[hyper]]
noun2parts_disting['millionaire']
noun2parts_disting['messiah']
#keep only 'distinguishing' parts
for ss in whole_ss:
    paths = ss.hypernym_paths()
    for p in paths:
        for i,hyper in enumerate(p):
            #skip self-reference
            if i == len(p) - 1:
                continue
            if hyper in whole_ss:
                print("removing %s parts from %s" % (ss2whole[hyper], ss2whole[ss]))
                noun2parts_disting[ss2whole[ss]] = noun2parts_disting[ss2whole[ss]] - noun2parts_disting[ss2whole[hyper]]
noun2parts_disting['grandson']
noun2parts_disting['apache']
wn.synsets('salesman').hypernym_paths()
wn.synsets('salesman')[0].hypernym_paths()
noun2parts['person']
'person' in whole_ss
len(whole_ss)
whole2ss = {whole:wn.synsets(whole)[0] for whole in noun2parts.keys()}
whole2ss['person']
whole_ss = set(whole2ss.values())
'person' in whole_ss
whole_ss
whole2ss['person']
ss = whole2ss['person']
ss2whole[ss]
len(ss2whole)
len(whole2ss)
ss2wholes = defaultdict(set)
for w,ss in whole2ss.items():
    ss2wholes[ss].add(w)
    
ss2wholes[ss]
ss
ss = whole2ss['person']
ss2wholes[ss]
noun2parts_disting = copy.deepcopy(noun2parts)
type(whole_ss)
wholes = set(whole2ss.keys())
#keep only 'distinguishing' parts
for whole in wholes:
    paths = whole2ss[whole].hypernym_paths()
    for p in paths:
        for i,hyper in enumerate(p):
            #skip self-reference
            if i == len(p) - 1:
                continue
            if hyper in whole_ss:
                for hyper_whole in ss2wholes[hyper]:
                    noun2parts_disting[whole] = noun2parts_disting[whole] - noun2parts_disting[hyper_whole]
noun2parts_disting['hotel']
noun2parts_disting['apache']
i = 0
for noun, parts in noun2parts_disting.items():
    if len(parts) > 7:
        print(i, noun, sorted(list(parts)[:7]))
        i += 1
        
wn.synsets('art')[0].hypernym_paths()
wn.synsets('soy')[0].hypernym_paths()
wn.synsets('carrot')[0].hypernym_paths()
wn.synsets('carrot')[0].hypernym_paths()[-1]
wn.synsets('carrot')[0].hypernym_paths()[0][-1]
ss2whole[wn.synsets('carrot')[0].hypernym_paths()[0][-2]]
ss2whole[wn.synsets('carrot')[0].hypernym_paths()[0][-3]]
ss2whole[wn.synsets('carrot')[0].hypernym_paths()[0][-4]]
ss2whole[wn.synsets('carrot')[0].hypernym_paths()[0][-5]]
ss2whole[wn.synsets('carrot')[0].hypernym_paths()[0][-6]]
ss2whole[wn.synsets('carrot')[0].hypernym_paths()[0][-7]]
i = 0
for noun, parts in noun2parts_disting.items():
    if len(parts) > 1:
        print(i, noun, sorted(list(parts)[:5]))
        i += 1
        
i = 0
for noun, parts in noun2parts_disting.items():
    if len(parts) > 1:
        print(i, noun, sorted(list(parts)[:10]))
        i += 1
        
wn.synsets('rue')[0].hypernym_paths()[0][-1]
wn.synsets('rue')[0].hypernym_paths()
from gensim.corpora import Dictionary
dset = [list(parts) for n, parts in noun2parts_disting.items()]
dset[0]
dset[1]
dset[:10]
dct = Dictionary(dset)
dct
corpus = [dct.doc2bow(line) for line in dset]
corpus
from gensim.model import TfidfModel
from gensim.models import TfidfModel
model = TfidfModel(corpus)
model
dir(model)
model.dfs
model.id2word
dir(model)
model = TfidfModel(corpus, id2word=dct, dictionary=dct)
dir(model)
model.id2word
model.dfs
dir(model)
dir(model.id2word)
model.id2word.id2token
model.id2word.keys
model.id2word.keys()
dir(model.id2word)
dct.id2word
dct.id2token
dct.id2token()
dct.token2id
id2token = {i:t for t,i in model.id2word.token2id.items()}
id2token
model.dfs
import operator
sorted(model.dfs, key=operator.itemgetter(1), reverse=True)[:10]
sorted(model.dfs.items(), key=operator.itemgetter(1), reverse=True)[:10]
sorted(model.dfs.items(), key=operator.itemgetter(1), reverse=True)[:20]
[id2token[i], c for i, c in sorted(model.dfs.items(), key=operator.itemgetter(1), reverse=True)[:20]]
[(id2token[i], c) for i, c in sorted(model.dfs.items(), key=operator.itemgetter(1), reverse=True)[:20]]
i = 0
for noun, parts in noun2parts_disting.items():
    if len(parts) > 7:
        print(i, noun, sorted(list(parts)[:7]))
        i += 1
        
noun2parts_disting['eel']
noun2parts_disting['fish']
[(id2token[i], c) for i, c in sorted(model.dfs.items(), key=operator.itemgetter(1), reverse=True)[:20]]
[(id2token[i], c) for i, c in sorted(model.dfs.items(), key=operator.itemgetter(1), reverse=True)[:30]]
[(id2token[i], c) for i, c in sorted(model.dfs.items(), key=operator.itemgetter(1), reverse=True)[:50]]
[(id2token[i], c) for i, c in sorted(model.dfs.items(), key=operator.itemgetter(1), reverse=True)[:100]]
[(id2token[i], c) for i, c in sorted(model.dfs.items(), key=operator.itemgetter(1), reverse=True)[:70]]
[(id2token[i], c) for i, c in sorted(model.dfs.items(), key=operator.itemgetter(1), reverse=True)[:60]]
[(id2token[i], c) for i, c in sorted(model.dfs.items(), key=operator.itemgetter(1), reverse=True)[:56]]
[(id2token[i], c) for i, c in sorted(model.dfs.items(), key=operator.itemgetter(1), reverse=True)[:59]]
[(id2token[i], c) for i, c in sorted(model.dfs.items(), key=operator.itemgetter(1), reverse=True)[:58]]
part2wholes = defaultdict(set)
for noun, partset in noun2parts_disting.items():
    for part in partset:
        part2wholes[part].add(noun)
        
part2wholes['aperture']
part2wholes['armature']
noun2parts_disting['eel']
noun2parts['fish']
noun2parts_disting['fish']
wn.synsets('fish')[0].hypernym_paths()
wn.synsets('eel')[0].hypernym_paths()
wn.synsets('eel')[0].hypernym_paths()[0][-2]
ss = wn.synsets('eel')[0].hypernym_paths()[0][-2]
dir(ss)
ss.unicode_repr
ss.unicode_repr()
ss.offset
ss.offset()
ss.common_hypernyms()
ss.tree()
ss.name()
noun2part_disting['snapper']
noun2parts_disting['snapper']
noun2parts_disting['halibut']
noun2parts_disting['salmon']
i = 0
for noun, parts in noun2parts_disting.items():
    if len(parts) > 1:
        print(i, noun, sorted(list(parts)[:7]))
        i += 1
        
i = 0
for noun, parts in noun2parts_disting.items():
    if len(parts) > 0:
        print(i, noun, sorted(list(parts)[:7]))
        i += 1
        
ss2wholes = defaultdict(set)
noun2parts['footage']
noun2parts['toast']
noun2parts = get_part_whole_relations_pwkb()
noun2parts['toast']
noun2parts_vg = get_part_whole_relations_vg()
noun2parts_vg['toast']
noun2parts_vg['toast'].intersection(noun2parts['toast'])
dfs = Counter()
from collections import Counter
dfs = Counter()
for noun, parts in noun2parts_disting.items():
    for part in parts:
        dfs[part] += 1
        
dfs
sorted(dfs.items(), key=operator.itemgetter(1), reverse=False)[:30]
sorted(dfs.items(), key=operator.itemgetter(1), reverse=True)[:30]
i = 0
for noun, parts in noun2parts_disting.items():
    if len(parts) > 1:
        print(i, noun, sorted(list(parts)[:7]))
        i += 1
        
noun2parts['broom']
noun2parts_disting['broom']
wn.synsets('broom')[0]
wn.synsets('broom')[0].hypernym_paths()
'broomstick' in unigram_concrete
unigram_concrete = get_frequent_concrete_nouns()
noun2parts['cheeseburger']
noun2parts_disting['cheeseburger']
'cheeseburger' in unigram_concrete_whole
'cheeseburger' in unigram_concrete_whole_lem
noun2parts['burger']
unigram_concrete_whole = unigram_concrete.intersection(set(noun2parts.keys()))
lem = WordNetLemmatizer()
unigram_concrete_whole_lem = set()
for cand in unigram_concrete_whole:
    unigram_concrete_whole_lem.add(lem.lemmatize(cand))
len(unigram_concrete_whole_lem)
'cheeseburger' in unigram_concrete_whole_lem
'sandwich' in unigram_concrete_whole_lem
wn.synsets('youth')
wn.synsets('youth')[2]
wn.synsets('youth')[1]
wn.synsets('youth')[1].part_meronyms()
wn.synsets('youth')[2].part_meronyms()
model
model = gensim.models.KeyedVectors.load_word2vec_format('/home/jamesm/commonsense/data/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('/home/jamesm/commonsense/data/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
model = gensim.models.KeyedVectors.load_word2vec_format('/home/jamesm/commonsense/data/word2vec/GoogleNews-vectors-negative300.bin', binary=True, limit=100000)
model.most_similar('such')
model.most_similar('first')
model.most_similar('other')
model.most_similar('only')
model.most_similar('actual')
model.most_similar('real')
model.most_similar('early')
wn.synsets('great')
wn.synsets('great')[0]
wn.synsets('great')[0].hypernym_paths()
wn.synsets('great')
wn.synsets('great')[1]
dir(wn.synsets('great')[1])
ss = wn.synsets('great')[1]
ss.examples()
ss.attributes
ss.attributes)_
ss.attributes()
dir(wn.synsets('great')[1])
ss.part_meronyms()
ss.hypernym_paths()
ss.tree()
ss.root_hypernyms()
wn.synsets('entire')
wn.synsets('entire')[1]
wn.synsets('entire')[1].hypernym_paths()
wn.synsets('grotesque')
wn.synsets('grotesque')[1]
wn.synsets('grotesque')[1].hypernym_paths()
wn.synsets('delicious')[1].hypernym_paths()
wn.synsets('tasty')[1].hypernym_paths()
wn.synsets('delicious')
a
a = [1, 2, 3, 4, 5]
cumsum(a)
np
np.cumsum(a)
np.insert(0,0,np.cumsum(a))
np.insert(0,np.cumsum(a))
np.insert(1,0,np.cumsum(a))
np.insert(0,1,np.cumsum(a))
import spacy
nlp = spacy.load('en')
dir(nlp)
nlp.pipeline()
nlp.pipeline
nlp.pipe_names()
nlp.pipe_names
nlp("person's hair")
nlp("person's hair.")
doc = nlp("person's hair.")
for tok in doc:
    print(tok.pos_)
    
tagger = Tagger(nlp.vocab)
from spacy.pipeline import Tagger
tagger = Tagger(nlp.vocab)
import spacy
from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
doc = nlp(u"What's happened to me? he thought. It wasn't a dream.")
print([t.text for t in doc])
doc.tokens
next(doc)
doc
dir(doc)
print(doc)
len([t for t in doc])
len(doc.split())
csv
with open('pwkb.txt')
concs = {}
conc_sds = {}
with open('/home/jamesm/commonsense/data/brysbaert/Concreteness_ratings_Brysbaert_et_al_BRM.txt') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        if row[-1] == "Noun":
            concs[row[0]] = float(row[2])
            conc_sds[row[0]] = float(row[3])
            
concs['human']
import random
random.choice(concs.keys())
random.choice(list(concs.keys()))
concs['puppet']
concs['person']
concs['guy']
concs['district']
concs['personality']
concs['soul']
concs['entity']
concs['abstraction']
concs['idea']
concs['spirituality']
conc_sds['district']
conc_sds['puppet']
conc_sds['spirituality']
conc_sds['idea']
conc_sds['guy']
concs['mouse']
conc_sds['mouse']
concs['concrete']
concs['cement']
conc_sds['cement']
sorted(concs.items(), key=operator.itemgetter(1), reverse=true)[:20]
sorted(concs.items(), key=operator.itemgetter(1), reverse=True)[:20]
concs['angelfish']
conc_sds['angelfish']
concs
concs['dirt']
br_nouns = set(concs.keys())
len(br_nouns)
len(unigram_concrete)
len(br_nouns - unigram_concrete)
br_nouns - unigram_concrete
unigram_concrete - br_nouns
br_nouns
part2whole['hardship']
br_nouns
concs
score2nouns = defaultdict(set)
for noun, score in concs.items():
    score2nouns[score].add(noun)
    
score2nouns[4.00]
score2nouns[3.50]
sum([len(score2nouns[score/100]) for score in range(400,500)])
sum([len(score2nouns[score/100]) for score in range(300,500)])
sum([len(score2nouns[score/100]) for score in range(400,500)])
score2nouns[4.00]
score2nouns[4.50]
sum([len(score2nouns[score/100]) for score in range(450,500)])
len([noun for noun in concs.keys() if concs[noun] - conc_sds[noun] > 4.0])
len([noun for noun in concs.keys() if concs[noun] - conc_sds[noun] > 3.5])
br_nouns = set([noun for noun in concs.keys() if concs[noun] - conc_sds[noun] > 3.5])
br_nouns
unigram_concrete - br_nouns
'sequin' in br_nouns
'crutch' in br_nouns
'trawler' in br_nouns
'sauerkraut' in concs.keys()
concs['sauerkraut']
conc_sds['sauerkraut']
len(br_nouns)
len(concs.keys())
len(set(br_nouns)+set(concs.keys()))
len(set(br_nouns).union(set(concs.keys())))
len(set(br_nouns).union(set(unigram_concrete)))
len(unigram_concrete)
br_nouns - unigram_concrete
len(unigram_concrete)
br_nouns
br_nouns - unigram_concrete
'vodka'
new = br_nouns - unigram_concrete
len(unigram_concrete_whole)
len(unigram_concrete)
unigram_concrete2 = unigram_concrete.union(br_nouns)
len(unigram_concrete2)
len(new)
unigram_concrete_whole2 = unigram_concrete2.intersection(set(noun2parts.keys()))
len(noun2parts.keys())
len(whole2parts)
whole2parts_vg = get_part_whole_relations_vg(lem)
whole2parts_pwkb = get_part_whole_relations_pwkb(lem)
from part_whole_candidates import get_part_whole_relations_pwkb
len(whole2parts_vg)
from part_whole_candidates import get_part_whole_relations_vg
whole2parts_vg = get_part_whole_relations_vg(lem)
len(whole2parts_vg)
whole2parts_pwkb = get_part_whole_relations_pwkb(lem)
len(whole2parts_pwkb)
whole2parts = defaultdict(set, {whole: whole2parts_vg[whole].union(whole2parts_pwkb[whole])                                 for whole in set(whole2parts_vg.keys()).union(set(whole2parts_pwkb.keys()))})
len(whole2parts)
unigram_concrete_whole = unigram_concrete.intersection(set(whole2parts.keys()))
len(unigram_concrete_whole)
unigram_concrete_whole2 = unigram_concrete2.intersection(set(whole2parts.keys()))
len(unigram_concrete_whole2)
unigram_concrete2 - unigram_concrete_whole
lem.lemmatize('assistants')
'assistant' in unigram_concrete_whole
br_nouns = set([lem.lemmatize(noun) for noun in concs.keys() if concs[noun] - conc_sds[noun] > args.concreteness])
br_nouns = set([lem.lemmatize(noun) for noun in concs.keys() if concs[noun] - conc_sds[noun] > 3.5])
len(br_nouns)
unigram_concrete2 = unigram_concrete.union(br_nouns)
len(unigram_concrete2)
new = br_nouns - unigram_concrete
len(new)
new
unigram_concrete_whole2 = unigram_concrete2.intersection(set(whole2parts.keys()))
len(unigram_concrete_whole2)
unigram_concrete2 - unigram_concrete_whole
unigram_concrete_whole2 - unigram_concrete_whole
whole2parts['yard']
whole2parts['yak']
unigram_concrete_whole2 - unigram_concrete_whole
whole2parts['wine']
whole2parts['wood']
whole2parts2 = {noun: set([part for part in partset if part in unigram_concrete_whole2]) for noun,partset in whole2parts2.items() if noun in unigram_concrete_whole2}
whole2parts2 = copy.deepcopy(whole2parts)
whole2parts2 = {noun: set([part for part in partset if part in unigram_concrete_whole2]) for noun,partset in whole2parts2.items() if noun in unigram_concrete_whole2}
len(whole2parts)
len(whole2parts2)
whole2parts = {noun: set([part for part in partset if part in unigram_concrete_whole]) for noun,partset in whole2parts.items() if noun in unigram_concrete_whole}
len(whole2parts)
whole2parts2['mold']
unigram_concrete_whole2 - unigram_concrete_whole
whole2parts2['volcano']
whole2parts2['tobacco']
whole2parts2['traffic']
whole2parts2['dustpan']
whole2parts2['dynamite']
unigram_concrete_whole2 - unigram_concrete_whole
whole2parts2['earplug']
whole2parts2['coupon']
whole2parts2['volleyball']
whole2parts2['wetsuit']
whole2parts2['vodka']
whole2parts2['smile']
whole2parts2['orchestra']
sum(len(whole2parts2.values()))
sum([len(s) for s in whole2parts2.values()])
whole2parts2['person']
whole2parts_vg['person']
whole2parts2['person']
