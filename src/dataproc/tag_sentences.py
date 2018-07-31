"""
    Given an input set of sentences, get POS-tags with spacy and get token match features (for DIIN)
"""
import argparse
import csv
import json

import nltk
from nltk.corpus import wordnet as wn 
import spacy
from tqdm import tqdm

stemmer = nltk.SnowballStemmer('english')

def is_exact_match(token1, token2):
    token1 = token1.lower()
    token2 = token2.lower()
    
    token1_stem = stemmer.stem(token1)

    if token1 == token2:
        return True
    
    for synsets in wn.synsets(token2):
        for lemma in synsets.lemma_names():
            if token1_stem == stemmer.stem(lemma):
                return True
    
    if token1 == "n't" and token2 == "not":
        return True
    elif token1 == "not" and token2 == "n't":
        return True
    elif token1_stem == stemmer.stem(token2):
        return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="_train name of sentence-style data")
    args = parser.parse_args()

    nlp = spacy.load('en')

    for fold in ['train', 'dev', 'test']:
        with open(args.file.replace('train', fold)) as f:
            with open(args.file.replace("train.csv", "%s_feats.jsonl" % fold), 'w') as of:
                r = csv.reader(f, delimiter='\t')
                #w = csv.writer(of, delimiter='\t')
                header = next(r)
                #header.extend(['context_pos', 'hypothesis_pos', 'context_match_hyp', 'hyp_match_context'])
                #w.writerow(header)
                for row in tqdm(r):
                    obj = {'whole': row[0], 'part': row[1], 'jj': row[2], 'sentence1': row[4], 'sentence2': row[3], 'label': row[5], 'bin_label': row[6]}

                    context_raw, hyp_raw = row[4], row[3]
                    context = nlp(context_raw)
                    hyp = nlp(hyp_raw)
                    context_pos = ' '.join([tok.tag_ for tok in context])
                    hyp_pos = ' '.join([tok.tag_ for tok in hyp])

                    c_h = set()
                    h_c = set()
                    for i, cword in enumerate(context):
                        #import pdb; pdb.set_trace()
                        for j, hword in enumerate(hyp):
                            if is_exact_match(cword.text, hword.text):
                                c_h.add(i)
                                h_c.add(j)
                    c_h = sorted(c_h)
                    h_c = sorted(h_c)

                    obj['sentence1_parse'] = context_pos
                    obj['sentence2_parse'] = hyp_pos
                    obj['sentence1_binary_parse'] = ' '.join([tok.text for tok in context])
                    obj['sentence2_binary_parse'] = ' '.join([tok.text for tok in hyp])
                    obj['sentence1_token_exact_match_with_s2'] = c_h
                    obj['sentence2_token_exact_match_with_s1'] = h_c

                    of.write(json.dumps(obj) + '\n')

                    #w.writerow([*row, context_pos, hyp_pos, ' '.join([str(n) for n in c_h]), ' '.join([str(n) for n in h_c])])

