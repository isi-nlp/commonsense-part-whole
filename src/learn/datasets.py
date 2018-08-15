import csv
import json

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset
import spacy

class TripleDataset(Dataset):
    def __init__(self, fname, binary, only_use, **kwargs):
        self.triples = pd.read_csv(fname)
        self.binary = binary
        self.word2ix = None
        if only_use == 'pw':
            self.words = ['whole', 'part']
        elif only_use == 'wjj':
            self.words = ['whole', 'jj']
        elif only_use == 'pjj':
            self.words = ['part', 'jj']
        else:
            self.words = ['whole', 'part', 'jj']

        #make sklearn build the vocab for me
        vectorizer = CountVectorizer(tokenizer=str.split)
        #self.triples['cat'] = self.triples.apply(lambda row: ' '.join([row['whole'], row['part'], row['jj']]), axis=1)
        self.triples['cat'] = self.triples.apply(lambda row: ' '.join([row[w] for w in self.words]), axis=1)
        feats = vectorizer.fit_transform(self.triples['cat'])
        self.word2ix = vectorizer.vocabulary_

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples.iloc[idx][self.words].tolist()
        lname = 'bin_label' if self.binary else 'label'
        return triple, self.triples.iloc[idx][lname]

class TripleImageDataset(TripleDataset):
    def __init__(self, fname, binary, only_use, **kwargs):
        super(TripleImageDataset, self).__init__(fname, binary, only_use)
        pws = set()
        for row in self.triples.itertuples():
            pws.add((row.whole, row.part))
        #also load image feature lookups
        self.pw2featw = {}
        self.pw2featp = {}
        with open('../../data/candidates/pw_img_avg_feats.jsonl') as f:
        #with open('../../data/candidates/pw_max_iou_feats.jsonl') as f:
            for line in f:
                obj = json.loads(line.strip())
                whole, part = obj['whole'], obj['part']
                if (whole, part) in pws:
                    self.pw2featw[(whole, part)] = obj['featw']
                    self.pw2featp[(whole, part)] = obj['featp']

    def __getitem__(self, idx):
        triple = self.triples.iloc[idx][self.words].tolist()
        pw = tuple(triple[:2])
        lname = 'bin_label' if self.binary else 'label'
        return triple, self.triples.iloc[idx][lname], self.pw2featw[pw], self.pw2featp[pw]

class DefinitionDataset(Dataset):
    def __init__(self, fname, binary, only_use, **kwargs):
        self.dataset = pd.read_csv(fname, delimiter='\t')
        self.binary = binary
        self.word2ix = None
        if only_use == 'pw':
            self.words = ['whole', 'part']
        elif only_use == 'wjj':
            self.words = ['whole', 'jj']
        elif only_use == 'pjj':
            self.words = ['part', 'jj']
        else:
            self.words = ['whole', 'part', 'jj']
        
        #make sklearn/spacy build the vocab for me
        self.nlp = spacy.load('en', disable=['tagger', 'parser', 'ner', 'textcat'])
        vectorizer = CountVectorizer(tokenizer=str.split)
        self.dataset['cat'] = self.dataset.apply(self._preprocess_dfns, axis=1)
        feats = vectorizer.fit_transform(self.dataset['cat'])
        self.word2ix = vectorizer.vocabulary_

    def _preprocess_dfns(self, row):
        joined = ' '.join([row[f'{w}_def'] for w in self.words])
        return ' '.join([tok.text for tok in self.nlp(joined)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dfns = self.dataset.iloc[idx][[f'{w}_def' for w in self.words]].tolist()
        lname = 'bin_label' if self.binary else 'label'
        return dfns, self.dataset.iloc[idx][lname]

class TripleRetrDataset(Dataset):
    def __init__(self, fname, binary, only_use, trip2embeds):
        trip2label = {}
        pws = set()
        for row in csv.reader(open(fname)):
            trip2label[tuple(row[:3])] = tuple(row[3:])
            pws.add(tuple(row[:2]))
        self.data = []
        self.labels = []
        self.trips = []
        for trip,embeds in trip2embeds.items():
            if trip not in trip2label:
                continue
            if trip[:2] not in pws:
                continue
            for embed in embeds: 
                self.data.append(embed)
                self.labels.append(tuple([int(x) for x in trip2label[trip]]))
                self.trips.append(trip)
        self.binary = binary
        self.word2ix = None
        if only_use == 'pw':
            self.words = ['whole', 'part']
        elif only_use == 'wjj':
            self.words = ['whole', 'jj']
        elif only_use == 'pjj':
            self.words = ['part', 'jj']
        else:
            self.words = ['whole', 'part', 'jj']

        #make sklearn build the vocab for me
        self.triples = pd.read_csv(fname)
        vectorizer = CountVectorizer(tokenizer=str.split)
        #self.triples['cat'] = self.triples.apply(lambda row: ' '.join([row['whole'], row['part'], row['jj']]), axis=1)
        self.triples['cat'] = self.triples.apply(lambda row: ' '.join([row[w] for w in self.words]), axis=1)
        feats = vectorizer.fit_transform(self.triples['cat'])
        self.word2ix = vectorizer.vocabulary_

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        l_ix = 1 if self.binary else 0
        return self.trips[idx], self.labels[idx][l_ix], self.data[idx], 


class TripleBboxDataset(TripleDataset):
    def __init__(self, fname, binary, only_use):
        super(TripleBboxDataset, self).__init__(fname, binary, only_use)
        data_path = os.path.dirname(fname)
        self.part_feats = json.load(open(os.path.join(data_path, 'part_feats.json')))
        self.whole_feats = json.load(open(os.path.join(data_path, 'whole_feats.json')))
        self.pw_feats = {pw: (feat if not np.isnan(feat) else 0.0) for pw, feat in json.load(open(os.path.join(data_path, 'pw_feats.json'))).items()}
        self._clean_nans()

    def _clean_nans(self):
        for feat in [self.part_feats, self.whole_feats]:
            for name, dct in feat.items():
                for featname, val in dct.items():
                    if np.isnan(val):
                        feat[name][featname] = 0.0

    def __getitem__(self, idx):
        triple, label = super(TripleBboxDataset, self).__getitem__(idx)
        whole, part, _ = triple
        bbox_feats = [self.whole_feats[whole]['avg_w'], self.whole_feats[whole]['avg_h'],\
                      self.part_feats[part]['avg_w'], self.part_feats[part]['avg_h'],\
                      self.pw_feats[','.join(triple[:2])]]
        return triple, label, bbox_feats


