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

def get_idxs(obj):
    part, whole, jj = obj['part'], obj['whole'], obj['jj']
    prem_tokenized = [tok.text.lower() for tok in nlp(obj['sentence1'])]
    hyp_tokenized = [tok.text.lower() for tok in nlp(obj['sentence2'])]
    whole_idx1, whole_idx2, part_idx1, part_idx2, jj_idx1, jj_idx2 = -1, -1, -1, -1, -1, -1
    if len(whole.split()) == 1:
        if whole == 't-shirt':
            #spacy is dumb
            if whole in obj['sentence1']:
                prem_tokenized = [tok for tok in nltk.word_tokenize(obj['sentence1'])]
                whole_idx1 = prem_tokenized.index(whole)
            if whole in obj['sentence2']:
                hyp_tokenized = [tok for tok in nltk.word_tokenize(obj['sentence2'])]
                whole_idx2 = hyp_tokenized.index(whole)
        else:
            try:
                whole_idx1 = prem_tokenized.index(whole)
            except:
                pass
            try:
                whole_idx2 = hyp_tokenized.index(whole)
            except:
                pass
    else:
        whole1, whole2 = whole.split()
        #prem
        try:
            idx1 = prem_tokenized.index(whole1)
        except:
            prem_tokenized = retokenize(prem_tokenized)
            try:
                idx1 = prem_tokenized.index(whole1)
            except:
                idx1 = -1
        try:
            idx2 = prem_tokenized.index(whole2)
        except:
            prem_tokenized = retokenize(prem_tokenized)
            try:
                idx2 = prem_tokenized.index(whole2)
            except:
                idx2 = -1
        whole_idx1 = (idx1, idx2)
        #hyp
        try:
            idx1 = hyp_tokenized.index(whole1)
        except:
            if whole_idx1 == -1:
                hyp_tokenized = retokenize(hyp_tokenized)
                try:
                    idx1 = hyp_tokenized.index(whole1)
                except:
                    idx1 = -1
        try:
            idx2 = hyp_tokenized.index(whole2)
        except:
            if whole_idx1 == -1:
                hyp_tokenized = retokenize(hyp_tokenized)
                try:
                    idx2 = hyp_tokenized.index(whole2)
                except:
                    idx2 = -1
        whole_idx2 = (idx1, idx2)
        #ugh
        if -1 in whole_idx1:
            whole_idx1 = -1
        if -1 in whole_idx2:
            whole_idx2 = -1

    try:
        jj_idx1 = prem_tokenized.index(jj)
    except:
        pass
    try:
        jj_idx2 = hyp_tokenized.index(jj)
    except:
        pass
    try:
        part_idx1 = prem_tokenized.index(part)
    except:
        pass
    try:
        part_idx2 = hyp_tokenized.index(part)
    except:
        pass
    return whole, part, jj, prem_tokenized, hyp_tokenized, [whole_idx1, whole_idx2, part_idx1, part_idx2, jj_idx1, jj_idx2]

def get_vecs(prem_embeds, hyp_embeds, idx, idxs1, idxs2):
    if type(idxs1[0]) is tuple:
        #combine multiword wholes
        whole_vec1 = (prem_embeds[idx][idxs1[0][0]].data + prem_embeds[idx][idxs1[0][1]].data) / 2
    elif idxs1[0] != -1:
        whole_vec1 = prem_embeds[idx][idxs1[0]].data
    if type(idxs2[0]) is tuple:
        #combine multiword wholes
        whole_vec2 = (hyp_embeds[idx][idxs2[0][0]].data + hyp_embeds[idx][idxs2[0][1]].data) / 2
    elif idxs2[0] != -1:
        whole_vec2 = hyp_embeds[idx][idxs2[0]].data
    else:
        whole_vec2 = whole_vec1
    if idxs1[0] == -1:
        whole_vec1 = whole_vec2
    #PART
    if idxs1[1] != -1:
        part_vec1 = prem_embeds[idx][idxs1[1]].data
    if idxs2[1] != -1:
        part_vec2 = hyp_embeds[idx][idxs2[1]].data
    else:
        part_vec2 = part_vec1
    if idxs1[1] == -1:
        part_vec1 = part_vec2
    #JJ
    if idxs1[2] != -1:
        jj_vec1 = prem_embeds[idx][idxs1[2]].data
    if idxs2[2] != -1:
        jj_vec2 = hyp_embeds[idx][idxs2[2]].data
    else:
        jj_vec2 = jj_vec1
    if idxs1[2] == -1:
        jj_vec1 = jj_vec2
    return whole_vec1, whole_vec2, part_vec1, part_vec2, jj_vec1, jj_vec2


options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

print("loading elmo")
elmo = Elmo(options_file, weight_file, 1, dropout=0)
print("done")

#just the tokenizer
nlp = spacy.load('en', disable=['tagger', 'parser', 'ner', 'textcat'])

batch_size = 32
with open('../../data/annotated/elmo_snli_contextualized3.data', 'w') as of:
    w = csv.writer(of)
    w.writerow(['whole', 'part', 'jj', 'vec'])
    #with open('../../data/annotated/snli_style_feats.jsonl') as f:
    with open('../../data/annotated/retrieved_feats.jsonl') as f:
        prem_batch = []
        prem_batch_idxs = [] #tuples
        hyp_batch = []
        hyp_batch_idxs = []
        wbatch = []
        pbatch = []
        jbatch = []
        for idx,line in tqdm(enumerate(f)):
            obj = json.loads(line.strip())
            whole, part, jj, prem_tokenized, hyp_tokenized, idxs = get_idxs(obj)
            whole_idx1, whole_idx2, part_idx1, part_idx2, jj_idx1, jj_idx2 = idxs
            
            prem_batch.append(prem_tokenized)
            prem_batch_idxs.append((whole_idx1, part_idx1, jj_idx1))

            hyp_batch.append(hyp_tokenized)
            hyp_batch_idxs.append((whole_idx2, part_idx2, jj_idx2))

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
                for idx,(idxs1, idxs2) in enumerate(zip(prem_batch_idxs, hyp_batch_idxs)):
                    #WHOLE
                    whole_vec1, whole_vec2, part_vec1, part_vec2, jj_vec1, jj_vec2 = get_vecs(prem_embeds, hyp_embeds, idx, idxs1, idxs2)
                    
                    whole_vec = (whole_vec1 + whole_vec2) / 2
                    part_vec = (part_vec1 + part_vec2) / 2
                    jj_vec = (jj_vec1 + jj_vec2) / 2

                    vec = np.concatenate([whole_vec, part_vec, jj_vec])
                    w.writerow([wbatch[idx], pbatch[idx], jbatch[idx]] + vec.tolist())

                #reset
                prem_batch = []
                prem_batch_idxs = [] #tuples
                hyp_batch = []
                hyp_batch_idxs = []
                wbatch = []
                pbatch = []
                jbatch = []
        #final write
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
