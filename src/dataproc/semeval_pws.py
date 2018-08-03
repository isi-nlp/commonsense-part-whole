import csv

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
 
lines = open('../../data/semeval2010task8/data.txt').readlines()
lem = WordNetLemmatizer()

pws = set()
for i in range(len(lines)//4):
    sent = lines[i*4]
    label = lines[i*4+1]
    sent = sent.split('\t')[1].strip()
    label = label.strip()
    e1 = sent.split('<e1>')[1].split('</e1>')[0].lower()
    e2 = sent.split('<e2>')[1].split('</e2>')[0].lower()
    if label.startswith('Component-Whole'):
        if label.endswith('(e1,e2)'):
            whole, part = e2, e1
        elif label.endswith('(e2,e1)'):
            whole, part = e1, e2
        if ' ' in whole:
            if len(wn.synsets(whole.replace(' ', '_'))) > 0:
                whole = lem.lemmatize(whole.replace(' ', '_'))
            elif len(wn.synsets(whole.replace(' ', ''))) > 0:
                whole = lem.lemmatize(whole.replace(' ', ''))
            else:
                print(whole.replace(' ', '_'), part)
                continue
        else:
            whole = lem.lemmatize(whole)
        if ' ' in part:
            if len(wn.synsets(part.replace(' ', '_'))) > 0:
                part = lem.lemmatize(part.replace(' ', '_'))
            elif len(wn.synsets(part.replace(' ', ''))) > 0:
                part = lem.lemmatize(part.replace(' ', ''))
            else:
                print(whole, part.replace(' ', '_'))
                continue
        else:
            part = lem.lemmatize(part)
        pws.add((whole, part))
        
with open('../data/nouns/semeval_pws.csv', 'w') as of:
    w = csv.writer(of)
    for whole, part in pws:
        w.writerow([whole.replace('_', ' '), part.replace('_', ' ')])
        
