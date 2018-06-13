"""
    Write the set of all nouns included in 'has' relations in the Visual Genome dataset
"""
from collections import Counter, defaultdict, namedtuple
import csv
import json
import operator
import random
import re

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

def tfidf(whole, part, subj_obj_imgs):
    """
        Compute the tfidf-like ratio (# images with (whole, part)) / (# images with part *as the part in a has relation*)
        If the ratio is too low, we will prune this (whole, part)
    """
    tf = len(subj_obj_imgs[(whole, part)])
    df = sum([len(imgs) for ((_, part_), imgs) in subj_obj_imgs.items() if part_ == part])
    return tf/df

def tfidf_abs(whole, part, subj_obj_imgs, abs_noun_counts):
    """
        Compute the tfidf-like ratio (# images with (whole, part)) / (# images with part *anywhere*)
        If the ratio is too low, we will prune this (whole, part)
    """
    tf = len(subj_obj_imgs[(whole, part)])
    df = abs_noun_counts[part]
    try:
        return tf/df
    except:
        import pdb; pdb.set_trace()

def get_name_and_id(rel, arg):
    if 'name' in rel[arg]:
        name = rel[arg]['name']
    elif 'names' in rel[arg]:
        name = rel[arg]['names'][0]
    return re.sub('\s\s+', ' ', name), rel[arg]['object_id']

def get_part_to_filtered_lookup(vgrels):
    ### spacy works much faster when you give it a lot of text at once
    ### so, here's an initial loop to read all part nouns, pos-tag and a filter them,
    ### and create a lookup. Also remove non-alphabetic chars
    print("making part to filtered lookup")
    print("filtering all part nouns")
    parts = set()
    for img in tqdm(vgrels):
        for rel in img['relationships']:
            if rel['predicate'] in ['has', 'has a', 'have', 'has an']:
                parts.add(get_name_and_id(rel, 'object')[0])
                parts.add(get_name_and_id(rel, 'subject')[0])
    parts = list(parts)
    inds = np.cumsum([len(p.split()) for p in parts])
    words = re.sub('\s\s+', ' ', " ".join(parts))
    print("pos tagging...")
    doc = nlp(words)
    part2filtered = {}
    cur_orig_part = []
    cur_filt = None
    ix = 0
    for i, t in enumerate(doc):
        if i < inds[ix]:
            cur_orig_part.append(t.text)
            if t.pos_ == 'NOUN':
                cur_filt = t.text
        elif i == inds[ix]:
            if cur_filt is not None:
                part2filtered[' '.join(cur_orig_part)] = re.sub('[^a-zA-Z]', '', cur_filt)
            ix += 1
            cur_orig_part = [t.text]
            cur_filt = t.text if t.pos_ == 'NOUN' else None
    return parts, part2filtered

if __name__ == "__main__":
    print("loading visual genome relationships...")
    vgrels = json.load(open('../data/visualgenome/relationships.json'))
    nlp = spacy.load('en_core_web_sm', parser=False, tagger=True, entity=False)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    BboxInfo = namedtuple('BboxInfo', ['id', 'px', 'py', 'pw', 'ph', 'wx', 'wy', 'ww', 'wh'])
    with open('../data/nouns/vg_has_nouns_min_3_details.tsv', 'w') as of:
        w = csv.writer(of, delimiter='\t')
        w.writerow(['subject', 'object', 'num_whole', 'num_part', 'num_pw', 'img1', 'img2', 'img3'])
        #how many total instances of subj has obj
        subj_obj_counts = Counter() 
        #how many images where subj had obj
        subj_obj_imgs = defaultdict(list)
        #how many times each noun appears at all (not just in a 'has' relationship)
        abs_noun_counts = Counter()

        parts, part2filtered = get_part_to_filtered_lookup(vgrels)
        
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

                #filter names
                if obj_name in part2filtered:
                    obj_name = part2filtered[obj_name]
                if subj_name in part2filtered:
                    subj_name = part2filtered[subj_name]

                if subj_name is None or obj_name is None:
                    continue

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

                    #subj_objs.add((subj, subj_ss, obj, obj_ss))
                    #subj_objs.append((subj, subj_ss, obj, obj_ss))
                    subj_obj_counts[(subj_name, obj_name)] += 1
                    if (subj_name, obj_name) not in added:
                        bbox = BboxInfo(img['image_id'], obj['x'], obj['y'], obj['w'], obj['h'], subj['x'], subj['y'], subj['w'], subj['h'])
                        subj_obj_imgs[(subj_name, obj_name)].append(bbox)
                        added.add((subj_name, obj_name))

        #print("calculating tfidfs...")
        #tfidfs = defaultdict(lambda: {})
        ##tfidfs_abs = defaultdict(lambda: {})
        #for (whole, part) in tqdm(subj_obj_imgs.keys()):
        #    tfidfs[whole][part] = tfidf(whole, part, subj_obj_imgs)
        #    #tfidfs_abs[whole][part] = tfidf_abs(whole, part, subj_obj_imgs, abs_noun_counts)


        whole_imgs = defaultdict(set)
        part_imgs = defaultdict(set)
        for (subj, obj), images in subj_obj_imgs.items():
            whole_imgs[subj].update(set([*images]))
            part_imgs[obj].update(set([*images]))

        print("writing part-whole pairs...")
        for (whole, part), images in subj_obj_imgs.items():
            if len(subj_obj_imgs[(whole, part)]) > 2:
                imgs = list(subj_obj_imgs[(whole, part)])
                random.shuffle(imgs)
                sample = [json.dumps(img._asdict()) for img in imgs[:3]]
                w.writerow([whole, part, len(whole_imgs[whole]), len(part_imgs[part]), len(subj_obj_imgs[(whole, part)]), *sample])
        #for whole, lookup in tfidfs.items():
        #    for part, score in lookup.items():
        #        #this set of requirements also implicitly requires a part to show up at least 3 times
        #        if len(subj_obj_imgs[(whole, part)]) > 1 and score > 0.1 and score != 1.0 and score != 0.5 and part != whole:
        #            w.writerow([whole, part, len(whole_imgs[whole]), len(part_imgs[part]), len(subj_obj_imgs[(whole, part)])])
