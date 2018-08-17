from collections import defaultdict
import json

import numpy as np
from tqdm import tqdm

pw2num = defaultdict(int)
pw2featw = defaultdict(lambda: np.zeros(2048))
pw2featp = defaultdict(lambda: np.zeros(2048))
with open('../../data/candidates/pw_img_feats2.jsonl') as f:
    for line in tqdm(f):
        obj = json.loads(line.strip())
        whole, part = obj['whole'], obj['part']
        pw2num[(whole, part)] += 1
        pw2featw[(whole, part)] += np.array(obj['featw'])
        pw2featp[(whole, part)] += np.array(obj['featp'])
            
for pw, num in pw2num.items():
    pw2featw[pw] /= num
    pw2featp[pw] /= num
    
with open('../../data/candidates/pw_img_avg_feats.jsonl', 'w') as of:
    for (whole, part) in tqdm(pw2featw.keys()):
        of.write(json.dumps({'whole': whole, 'part': part, 'featw': pw2featw[(whole, part)].tolist(), 'featp': pw2featp[(whole, part)].tolist()}) + '\n')
