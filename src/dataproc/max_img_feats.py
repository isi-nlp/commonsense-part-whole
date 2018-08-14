from collections import defaultdict
import json

from tqdm import tqdm

pw2max = defaultdict(int)
pw2vecs = {}
with open('../../data/candidates/pw_img_feats2.jsonl') as of:
    for line in tqdm(of):
        obj = json.loads(line.strip())
        whole, part = obj['whole'], obj['part']
        if obj['iouw'] + obj['ioup'] > pw2max[(whole, part)]:
            pw2max[(whole, part)] = obj['iouw'] + obj['ioup']
            pw2vecs[(whole, part)] = (obj['featw'], obj['featp'])
            
with open('../../data/candidates/pw_max_iou_feats.jsonl', 'w') as of:
    for (whole, part), (featw, featp) in tqdm(pw2vecs.items()):
        of.write(json.dumps({'whole': whole, 'part': part, 'featw': featw, 'featp': featp}) + '\n')
