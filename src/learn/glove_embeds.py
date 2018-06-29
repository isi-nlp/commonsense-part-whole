"""
    Extract glove embeds for the given vocab
"""

import csv, json, sys

from tqdm import tqdm

vocab = set([line.strip() for line in open(sys.argv[1])])
seen = set()
embeds = {}
with open(sys.argv[2]) as f:
    for line in tqdm(f):
        row = line.split()
        if row[0] in vocab:
            embeds[row[0]] = [float(r) for r in row[1:]]
            seen.add(row[0])
        if len(vocab.difference(seen)) == 0:
            break

with open(sys.argv[3], 'w') as of:
    json.dump(embeds, of, indent=1)
