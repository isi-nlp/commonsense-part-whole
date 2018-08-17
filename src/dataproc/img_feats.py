from collections import defaultdict
import json
import numpy as np
import re
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm

import base64
import csv
import sys 

def iou(bboxp, bboxw):
    ict_left = max(bboxp[0], bboxw[0])
    ict_right = min(bboxp[2], bboxw[2])
    ict_top = max(bboxp[1], bboxw[1])
    ict_bottom = min(bboxp[3], bboxw[3])
    if ict_left > ict_right or ict_top > ict_bottom:
        return 0

    ict = (ict_right - ict_left) * (ict_bottom - ict_top)
    union = (bboxp[2]-bboxp[0])*(bboxp[3]-bboxp[1]) + (bboxw[2]-bboxw[0])*(bboxw[3]-bboxw[1]) - ict
    return ict/union

def get_name_and_id(rel, arg):
    if 'name' in rel[arg]:
        name = rel[arg]['name']
    elif 'names' in rel[arg]:
        name = rel[arg]['names'][0]
    return re.sub('\s\s+', ' ', name), rel[arg]['object_id']


#LOAD IMAGE FEATURES
csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infile = '../../lib/bottom-up-attention/test2014_resnet101_faster_rcnn_genome.tsv.0'

print("reading image data")
in_data = {}
for i in range(2):
    with open(infile.replace('.tsv.0', '.tsv.%d' % i), "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for idx,item in tqdm(enumerate(reader)):
            try:
                item['image_id'] = int(item['image_id'])
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])
                item['num_boxes'] = int(item['num_boxes'])
                for field in ['boxes', 'features']:
                    item[field] = np.frombuffer(base64.decodestring(item[field]),
                          dtype=np.float32).reshape((item['num_boxes'],-1))
                in_data[item['image_id']] = item
            except:
                #bail
                continue

print("loading relationships")
rels = json.load(open('../../data/visualgenome/relationships.json'))
haspreds = ['has', 'has a', 'have', 'has an']
partpreds = ['part of', 'are part of', 'part of a', 'part', 'a part of', 'a part of a', 'are part of a']
allpreds = haspreds + partpreds

pws = set([tuple(row[:2]) for row in csv.reader(open('../../data/annotated/full_all.csv'))])

lem = WordNetLemmatizer()
pw2feats = defaultdict(list)
print("attaching image data to part-wholes")
for img in tqdm(rels):
    img_id = int(img['image_id'])
    #skip missing image features
    if img_id not in in_data:
        continue
    bboxes, feats = in_data[img_id]['boxes'], in_data[img_id]['features']
    for rel in img['relationships']:
        pred = rel['predicate']
        if pred in allpreds:
            #get whole and part name and bbox as appropriate
            if pred in haspreds:
                whole, _ = get_name_and_id(rel, 'subject')
                wholej = rel['subject']
                part, _ = get_name_and_id(rel, 'object')
                partj = rel['object']
                wbbox = [wholej['x'], wholej['y'], wholej['x']+wholej['w'], wholej['x']+wholej['h']]
                pbbox = [partj['x'], partj['y'], partj['x']+partj['w'], partj['x']+partj['h']]
            elif pred in partpreds:
                whole, _ = get_name_and_id(rel, 'object')
                wholej = rel['object']
                part, _ = get_name_and_id(rel, 'subject')
                partj = rel['subject']
                wbbox = [wholej['x'], wholej['y'], wholej['x']+wholej['w'], wholej['x']+wholej['h']]
                pbbox = [partj['x'], partj['y'], partj['x']+partj['w'], partj['x']+partj['h']]

            whole = lem.lemmatize(whole.replace(' ', '_')).replace('_', ' ')
            part = lem.lemmatize(part.replace(' ', '_')).replace('_', ' ')
            #if PW in this image is in data, find its best bbox
            if (whole, part) in pws:
                max_iouw = 0
                featw = None
                max_ioup = 0
                featp = None
                #match to best bboxes
                for bbox,feat in zip(bboxes, feats):
                    iouw = iou(wbbox, bbox)
                    ioup = iou(pbbox, bbox)
                    if iouw > max_iouw:
                        max_iouw = iouw
                        featw = feat
                    if ioup > max_ioup:
                        max_ioup = ioup
                        featp = feat
                if featw is not None and featp is not None:
                    if (whole, part) not in pw2feats and (len(pw2feats)+1) % 100 == 0:
                        print('%d part-wholes covered' % len(pw2feats))
                    pw2feats[(whole, part)].append({'featw': featw.tolist(), 'featp': featp.tolist(), 'iouw': max_iouw, 'ioup': max_ioup})

print("writing out features")
with open('../../data/candidates/pw_img_feats2.jsonl', 'w') as of:
    for (whole, part), insts in tqdm(pw2feats.items()):
        for inst in insts:
            inst['whole'] = whole
            inst['part'] = part
            of.write(json.dumps(inst) + '\n')
