"""
    Write the set of all nouns included in 'has' relations in the Visual Genome dataset, along with some image metadata
"""
from collections import Counter, defaultdict, namedtuple
import csv
import json
import operator
import random
import re

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import spacy
from spacy.tokens import Doc
from tqdm import tqdm

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

def get_name_and_id(rel, arg):
    if 'name' in rel[arg]:
        name = rel[arg]['name']
    elif 'names' in rel[arg]:
        name = rel[arg]['names'][0]
    return re.sub('\s\s+', ' ', name), rel[arg]['object_id']

if __name__ == "__main__":
    print("loading visual genome relationships...")
    vgrels = json.load(open('../../data/visualgenome/relationships.json'))
    nlp = spacy.load('en_core_web_sm', parser=False, tagger=True, entity=False)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    descriptors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black', 'white', 'gray', 'grey', 'brown', 'pink', 'front', 'back', 'left', 'right']
    has_descriptor = lambda name: any([desc in name for desc in descriptors])
    lem = WordNetLemmatizer()

    BboxInfo = namedtuple('BboxInfo', ['id', 'px', 'py', 'pw', 'ph', 'wx', 'wy', 'ww', 'wh'])
    with open('../../data/nouns/vg_min_3_wat.tsv', 'w') as of:
        w = csv.writer(of, delimiter='\t')
        w.writerow(['subject', 'object', 'num_whole', 'num_part', 'num_pw', 'img1', 'img2', 'img3'])
        #how many total instances of subj has obj
        subj_obj_counts = Counter() 
        #how many images where subj had obj
        subj_obj_imgs = defaultdict(list)
        #how many times each noun appears at all (not just in a 'has' relationship)
        abs_noun_counts = Counter()

        print("extracting parts and wholes from 'has' relations...")
        for img in tqdm(vgrels):
            added = set()
            #this will set the canonical name for this object in this image to avoid problems down the line
            img_id2name = {}
            for rel in img['relationships']:
                obj = rel['object']
                subj = rel['subject']
                subj_name, subj_id = get_name_and_id(rel, 'subject')
                obj_name, obj_id = get_name_and_id(rel, 'object')

                if obj_name is None or subj_name is None:
                    continue
                #other degenerate cases
                if '.' in obj_name or '.' in subj_name:
                    continue
                #if MWE
                if ' ' in obj_name:
                    #check if it's a common MWE in wordnet
                    if len(wn.synsets(obj_name)) == 0:
                        #also check without the space
                        if len(wn.synsets(''.join(obj_name.split()))) == 0:
                            if has_descriptor(obj_name):
                                words = obj_name.split()
                                if len(words) == 2:
                                    obj_name = words[-1]
                                else:
                                    #not dealing with this nonsense
                                    continue
                            else:
                                continue
                        else:
                            #if it's in wordnet only without the space, fix the name
                            obj_name = ''.join(obj_name.split())
                #if MWE
                if ' ' in subj_name:
                    #check if it's a common MWE in wordnet
                    if len(wn.synsets(subj_name.replace(' ', '_'))) == 0:
                        #also check without the space
                        if len(wn.synsets(''.join(subj_name.split()))) == 0:
                            if has_descriptor(subj_name):
                                words = subj_name.split()
                                if len(words) == 2:
                                    subj_name = words[-1]
                                else:
                                    #not dealing with this nonsense
                                    continue
                            else:
                                continue
                        else:
                            #if it's in wordnet only without the space, fix the name
                            subj_name = ''.join(subj_name.split())
                #lemmatize
                old_subj = subj_name
                subj_name = lem.lemmatize(subj_name.replace(' ', '_'))
                obj_name = lem.lemmatize(obj_name.replace(' ', '_'))

                #add counts for subj/obj
                if subj_id not in img_id2name:
                    img_id2name[subj_id] = subj_name
                    abs_noun_counts[subj_name] += 1
                else:
                    #object has been accounted for, use its canonical name
                    subj_name = img_id2name[subj_id]
                if obj_id not in img_id2name:
                    img_id2name[obj_id] = obj_name
                    abs_noun_counts[obj_name] += 1
                else:
                    #object has been accounted for, use its canonical name
                    obj_name = img_id2name[obj_id]

                if rel['predicate'] in ['has', 'has a', 'have', 'has an']:
                    #subject synset
                    if len(rel['subject']['synsets']) > 0:
                        subj_ss = rel['subject']['synsets'][0]
                    else:
                        subj_ss = None

                    #object synset
                    if len(rel['object']['synsets']) > 0:
                        obj_ss = rel['object']['synsets'][0]
                    else:
                        obj_ss = None

                    subj_obj_counts[(subj_name, obj_name)] += 1
                    if (subj_name, obj_name) not in added:
                        bbox = BboxInfo(img['image_id'], obj['x'], obj['y'], obj['w'], obj['h'], subj['x'], subj['y'], subj['w'], subj['h'])
                        subj_obj_imgs[(subj_name, obj_name)].append(bbox)
                        added.add((subj_name, obj_name))

        whole_imgs = defaultdict(set)
        part_imgs = defaultdict(set)
        for (subj, obj), images in subj_obj_imgs.items():
            whole_imgs[subj].update(set([*images]))
            part_imgs[obj].update(set([*images]))

        print("writing part-whole pairs...")
        for (whole, part), images in subj_obj_imgs.items():
            if len(subj_obj_imgs[(whole, part)]) >= 3:
                imgs = list(subj_obj_imgs[(whole, part)])
                random.shuffle(imgs)
                sample = [json.dumps(img._asdict()) for img in imgs[:3]]
                w.writerow([whole, part, len(whole_imgs[whole]), len(part_imgs[part]), len(subj_obj_imgs[(whole, part)]), *sample])
