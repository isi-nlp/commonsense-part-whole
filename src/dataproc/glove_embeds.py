"""
    Extract glove embeds for the given vocab
"""

import csv, json, sys

from tqdm import tqdm

vocab = set([line.strip() for line in open(sys.argv[1])])
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

with open(sys.argv[2], 'w') as of:
    json.dump(embeds, of, indent=1)
