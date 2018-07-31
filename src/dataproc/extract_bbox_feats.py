import csv, re, json
from collections import defaultdict

from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
from tqdm import tqdm

def get_name_and_id(rel, arg):
    if 'name' in rel[arg]:
        name = rel[arg]['name']
    elif 'names' in rel[arg]:
        name = rel[arg]['names'][0]
    return re.sub('\s\s+', ' ', name), rel[arg]['object_id']

def iou(bboxp, bboxw):
    ict_left = max(bboxp[0], bboxw[0])
    ict_right = min(bboxp[0]+bboxp[2], bboxw[0]+bboxw[2])
    ict_top = max(bboxp[1], bboxw[1])
    ict_bottom = min(bboxp[1]+bboxp[3], bboxw[1]+bboxw[3])
    if ict_left > ict_right or ict_top > ict_bottom:
        return 0
    
    ict = (ict_right - ict_left) * (ict_bottom - ict_top)
    union = bboxp[2]*bboxp[3] + bboxw[2]*bboxw[3] - ict
    return ict/union

rels = json.load(open('../../data/visualgenome/relationships.json'))
metadata = json.load(open('../../data/visualgenome/image_data.json'))
id2dims = {img['id']:(img['height'], img['width']) for img in metadata}

haspreds = ['has', 'has a', 'have', 'has an']
partpreds = ['part of', 'are part of', 'part of a', 'part', 'a part of', 'a part of a', 'are part of a']
allpreds = haspreds + partpreds

trips = set([tuple(row[:3]) for row in csv.reader(open('../../data/annotated/full_all.csv'))])
wholes = set([w for w,p,j in trips])
parts = set([p for w,p,j in trips])
pws = set([(w,p) for w,p,j in trips])

lem = WordNetLemmatizer()
part2widths = defaultdict(list)
part2heights = defaultdict(list)
whole2widths = defaultdict(list)
whole2heights = defaultdict(list)
pw2overlap = defaultdict(list)

partfeats = {part: {'avg_w': np.mean(part2widths[part]), 'avg_h': np.mean(part2heights[part])} for part in parts}
wholefeats = {whole: {'avg_w': np.mean(whole2widths[whole]), 'avg_h': np.mean(whole2heights[whole])} for whole in wholes}
pwfeats = {','.join(pw): np.mean(pw2overlap[pw]) for pw in pws}

with open('../../data/annotated/part_feats.json', 'w') as of:
    json.dump(partfeats, of)
    
with open('../../data/annotated/whole_feats.json', 'w') as of:
    json.dump(wholefeats, of)
    
with open('../../data/annotated/pw_feats.json', 'w') as of:
    json.dump(pwfeats, of)
    
for img in tqdm(rels):
    for rel in img['relationships']:
        pred = rel['predicate']
        if pred in allpreds:
            if pred in haspreds:
                whole, _ = get_name_and_id(rel, 'subject')
                wholej = rel['subject']
                part, _ = get_name_and_id(rel, 'object')
                partj = rel['object']
            elif pred in partpreds:
                whole, _ = get_name_and_id(rel, 'object')
                wholej = rel['object']
                part, _ = get_name_and_id(rel, 'subject')
                partj = rel['subject']
            whole = lem.lemmatize(whole.replace(' ', '_')).replace('_', ' ')
            part = lem.lemmatize(part.replace(' ', '_')).replace('_', ' ')
            if (whole, part) in pws:
                part2widths[part].append(partj['w']/id2dims[img['image_id']][1])
                part2heights[part].append(partj['h']/id2dims[img['image_id']][0])
                whole2widths[whole].append(wholej['w']/id2dims[img['image_id']][1])
                whole2heights[whole].append(wholej['h']/id2dims[img['image_id']][0])
                
                pw2overlap[(whole, part)].append(iou((partj['x'], partj['y'], partj['w'], partj['h']), (wholej['x'], wholej['y'], wholej['w'], wholej['h'])))
