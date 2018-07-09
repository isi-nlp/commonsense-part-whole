"""
    Given an input set of sentences, get POS-tags with spacy and get token match features
"""
import argparse

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
            with open(args.file.replace("train", "%s_feats" % fold), 'w') as of:
                r = csv.reader(f, delimiter='\t')
                w = csv.writer(of, delimiter='\t')
                header = next(r)
                header.extend(['context_pos', 'hypothesis_pos', 'context_match_hyp', 'hyp_match_context'])
                w.writerow(header)
                for row in tqdm(r):
                    context_raw, hyp_raw = row[3], row[4]

                    context = nlp(context_raw)
                    hyp = nlp(hyp_raw)
                    context_pos = ' '.join([' '.join((tok.tag_, tok.text)) for tok in context])
                    hyp_pos = ' '.join([' '.join((tok.tag_, tok.text)) for tok in hyp])

                    c_h = []
                    h_c = []
                    for i, cword in enumerate(context_raw):
                        for j, hword in enumerate(hyp_raw):
                            if is_exact_match(cword, hword):
                                c_h.append(j)
                                h_c.append(i)

                    w.writerow([*row, context_pos, hyp_pos, ' '.join([str(n) for n in c_h]), ' '.join([str(n) for n in h_c])])

