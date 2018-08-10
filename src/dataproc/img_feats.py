# coding: utf-8
import json
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer

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
infile = '../test2014_resnet101_faster_rcnn_genome.tsv.0'

in_data = {}
with open(infile, "r+b") as tsv_in_file:
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
    for item in reader:
        item['image_id'] = int(item['image_id'])
        item['image_h'] = int(item['image_h'])
        item['image_w'] = int(item['image_w'])
        item['num_boxes'] = int(item['num_boxes'])
        for field in ['boxes', 'features']:
            item[field] = np.frombuffer(base64.decodestring(item[field]),
                  dtype=np.float32).reshape((item['num_boxes'],-1))
        in_data[item['image_id']] = item

rels = json.load(open('../../data/visualgenome/relationships.json'))
haspreds = ['has', 'has a', 'have', 'has an']
partpreds = ['part of', 'are part of', 'part of a', 'part', 'a part of', 'a part of a', 'are part of a']
allpreds = haspreds + partpreds

lem = WordNetLemmatizer()
pw2feats = {}
for img in rels:
    img_id = img['image_id']
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
            if (whole, part) in pws:
                max_iouw = 0
                featw = None
                max_ioup = 0
                featp = None
                #match to best bbox
                for bbox,feat in zip(bboxes, feats):
                    iouw = iou(wbbox, bbox)
                    ioup = iou(pbbox, bbox)
                    if iouw > max_iouw:
                        max_iouw = iouw
                        featw = feat
                    if ioup > max_ioup:
                        max_ioup = ioup
                        featp = feat
                if max_bbox is not None:
                    mbboxes.append(max_bbox)
                    names.append(obj['names'][0])
                    
