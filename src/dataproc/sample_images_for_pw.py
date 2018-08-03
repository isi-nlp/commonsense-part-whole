"""
    Retrieve 3 random image ids for a sample of part-whole relations, to be uploaded to S3 for mturk HIT
"""
import csv, json, random
from collections import defaultdict
from tqdm import tqdm

print("loading vg relations")
vgrels = json.load(open('../../data/visualgenome/relationships.json'))

pws = set([(row[0], row[1]) for row in csv.reader(open('../../data/nouns/vg_has_nouns_min_3.csv'))])
pw2imgs = defaultdict(set)
for img in tqdm(vgrels):
    for rel in img['relationships']:
        if rel['predicate'] in ['has', 'has a', 'have', 'has an']:
            part = rel['object']
            whole = rel['subject']
            if (whole, part) in pws:
                pw2imgs[(whole, part)].add(img['image_id'])

with open('../../data/nouns/min_3_sample_imgs.csv', 'w') as of:
    w = csv.writer(of)
    for (whole, part), imgs in pw2imgs.items():
        imgs = list(imgs)
        random.shuffle(imgs)
        sample = imgs[:3]
        w.writerow([whole, part, *sample])
