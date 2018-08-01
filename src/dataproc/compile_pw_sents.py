# coding: utf-8
from collections import defaultdict
import json
import os

sents = defaultdict(set)
DATA = '/auto/nlg-05/jgm_234/commonsense-part-whole/data/'
GIGA = os.path.join(DATA, 'gigaword5-treebank')
UMBC = os.path.join(DATA, 'umbc-webbased-parsed')
for dr in [GIGA, UMBC]:
    for mode in ['of', 'poss']:
        ds = 'giga' if dr == GIGA else 'umbc'
        fn = f'{dr}/{mode}.{ds}.sents.json'
        print(f"Loading {fn}")
        fsents = json.load(open(fn))
        for pw,snts in fsents.items():
            sents[pw].update(set(snts))
    
#rework dict
print(f"part wholes covered: {len(sents)}")
obj = defaultdict(lambda: defaultdict(list))
for pw,sents in sents.items():
    whole, part = pw.split(',')
    obj[whole][part] = list(sents)
with open(f'{DATA}/candidates/pw_sents.json', 'w') as of:
    json.dump(obj, of)
