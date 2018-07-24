"""
    Use ELMo to get *contextualized* word vectors for part, whole, and adjective. These will then be used as input to MLP.
    TODO: Also try training end to end?
"""
import csv
import json

from allennlp.modules.elmo import Elmo, batch_to_ids
import nltk
import numpy as np
import spacy
from tqdm import tqdm

def retokenize(toklist):
    return [token for retoked in [nltk.word_tokenize(tok) for tok in toklist] for token in retoked]

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

print("loading elmo")
elmo = Elmo(options_file, weight_file, 1, dropout=0)

#just the tokenizer
nlp = spacy.load('en', disable=['tagger', 'parser', 'ner', 'textcat'])

batch_size = 32
with open('../../data/annotated/elmo_snli_contextualized.data', 'w') as of:
    w = csv.writer(of)
    w.writerow(['whole', 'part', 'jj', 'vec'])
    with open('../../data/annotated/snli_style_feats.jsonl') as f:
        prem_batch = []
        prem_batch_idxs = [] #tuples
        hyp_batch = []
        hyp_batch_idxs = []
        wbatch = []
        pbatch = []
        jbatch = []
        for idx,line in tqdm(enumerate(f)):
            obj = json.loads(line.strip())
            part, whole, jj = obj['part'], obj['whole'], obj['jj']
            prem_tokenized = [tok.text.lower() for tok in nlp(obj['sentence1'])]
            if len(whole.split()) == 1:
                whole_idx = prem_tokenized.index(whole)
            else:
                whole1, whole2 = whole.split()
                try:
                    idx1 = prem_tokenized.index(whole1)
                except:
                    prem_tokenized = retokenize(prem_tokenized)
                    idx1 = prem_tokenized.index(whole1)
                try:
                    idx2 = prem_tokenized.index(whole2)
                except:
                    prem_tokenized = retokenize(prem_tokenized)
                    idx2 = prem_tokenized.index(whole2)
                whole_idx = (idx1, idx2)
            jj_idx = prem_tokenized.index(jj)
            prem_batch.append(prem_tokenized)
            prem_batch_idxs.append((whole_idx, jj_idx))

            hyp_tokenized = [tok.text.lower() for tok in nlp(obj['sentence2'])]
            part_idx = hyp_tokenized.index(part)
            hyp_batch.append(hyp_tokenized)
            hyp_batch_idxs.append(part_idx)

            wbatch.append(whole)
            pbatch.append(part)
            jbatch.append(jj)

            if len(prem_batch) == batch_size:
                prem_char_ids = batch_to_ids(prem_batch)
                prem_embeds = elmo(prem_char_ids)
                prem_embeds = prem_embeds['elmo_representations'][0]

                hyp_char_ids = batch_to_ids(hyp_batch)
                hyp_embeds = elmo(hyp_char_ids)
                hyp_embeds = hyp_embeds['elmo_representations'][0]
                for idx,((iw, ij), ip) in enumerate(zip(prem_batch_idxs, hyp_batch_idxs)):
                    if type(iw) is tuple:
                        #combine multiword wholes
                        whole_vec = (prem_embeds[idx][iw[0]].data + prem_embeds[idx][iw[1]].data) / 2
                    else:
                        whole_vec = prem_embeds[idx][iw].data
                    vec = np.concatenate([whole_vec , hyp_embeds[idx][ip].data, prem_embeds[idx][ij].data])
                    w.writerow([wbatch[idx], pbatch[idx], jbatch[idx]] + vec.tolist())

                #reset
                prem_batch = []
                prem_batch_idxs = [] #tuples
                hyp_batch = []
                hyp_batch_idxs = []
