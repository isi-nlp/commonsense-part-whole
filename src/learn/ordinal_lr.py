"""
    Perform ordinal regression over pre-trained word vectors
"""
import argparse, csv, json

import mord
import numpy as np

from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error

def combine_embeds(triple, embeds):
    #just some hard coding...
    comb_embeds = []
    if ' ' in triple[0]:
        comb_embeds.append((embeds[0]+embeds[1])/2)
        if ' 'in triple[1]:
            #this doesn't happen :)
            pass
        comb_embeds.extend(embeds[2:])
    elif ' ' in triple[1]:
        comb_embeds.append(embeds[0])
        comb_embeds.append((embeds[1]+embeds[2])/2)
        comb_embeds.append(embeds[3])
    else:
        comb_embeds = embeds
    return comb_embeds

def process_data(fn, fold):
    X, Y = [], []
    with open(fn.replace('train', fold)) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            triple = tuple(row[:3])
            vecs = []
            for comp in triple:
                for c in comp.split():
                    vecs.append(np.array(w2v[c]))
            #import ipdb; ipdb.set_trace()
            vecs = combine_embeds(triple, vecs)
            X.append(np.concatenate(vecs))
            Y.append(int(row[4 if args.binary else 3]))

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='path to train file')
parser.add_argument("embed_file", type=str, help="path to embeddings file")
parser.add_argument('embed_type', choices=['elmo', 'glove', 'word2vec'], help='type of pretrained embedding to use')
parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs (default: 50)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
parser.add_argument('--binary', action='store_const', const=True, help='flag to predict binary labels instead of ordinal labels')
args = parser.parse_args()

#load embeddings
w2v = {}
if args.embed_type == 'elmo':
    h5 = h5py.File(args.embed_file, 'r')
    w2v = {word:vec.value[0] for word,vec in h5.items()}
elif args.embed_type == 'glove':
    with open(args.embed_file) as f:
        w2v = json.load(f)

X_train, Y_train = process_data(args.file, 'train')
X_dev, Y_dev = process_data(args.file, 'dev')
X_test, Y_test = process_data(args.file, 'test')

for clfclass in [mord.LogisticAT, mord.LogisticIT, mord.LogisticSE, mord.LAD, mord.OrdinalRidge]:
    clf = clfclass()
    print(clf)

    print("Training...")
    clf.fit(X_train, Y_train)
    #print(f"iterations: {clf.n_iter_}")

    preds = clf.predict(X_dev)

    print(f"acc: {accuracy_score(Y_dev, preds)}")
    print(f"prec: {precision_score(Y_dev, preds, average='weighted')}")
    print(f"rec: {recall_score(Y_dev, preds, average='weighted')}")
    print(f"f1: {f1_score(Y_dev, preds, average='weighted')}")
    print(f"mse: {mean_squared_error(Y_dev, preds)}")
    print(f"corr: {spearmanr(Y_dev, preds)[0]}")

