"""
    Given a file of triples, predicts the labels from:
    - most common overall class
    - most common class for given part (default to most common overall if unseen)
    - most common class for given whole (default to most common overall if unseen)
    - statistics from Google n-grams on the frequency of JJ amod PART
"""
import argparse, csv, itertools, operator
from collections import Counter, defaultdict, OrderedDict

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error

def get_preds_by_ngram_stats(fname, binary=False):
    print("reading noun-adjective counts from google dependency n-grams...")
    with open('../../data/ngrams/nn-jj-amod/nn.jj.amod.all') as f:
        r = csv.reader(f, delimiter=' ')
        noun2jjs = {}
        for row in r:
            noun = row[0]
            jjs = OrderedDict()
            for i in range(1,len(row),2):
                jjs[row[i]] = int(row[i+1])
            noun2jjs[noun] = jjs

    dev_fname = fname.replace('train', 'dev')
    test_fname = fname.replace('train', 'test')
    return _preds_for_file(noun2jjs, dev_fname, binary), _preds_for_file(noun2jjs, test_fname, binary)

def _preds_for_file(noun2jjs, fname, binary):
    with open(fname) as f:
        r = csv.reader(f)
        #header
        next(r)
        triples = []
        preds = []
        for row in r:
            triples.append(tuple(row[:3]))
            part, jj = row[1], row[2]
            if part in noun2jjs:
                jj_order = [jj for jj, count in sorted(noun2jjs[part].items(), key=operator.itemgetter(1), reverse=True)]
                if jj in jj_order:
                    norm_rank = jj_order.index(jj) / len(jj_order)
                    #guess unrelated by default (if never found)
                    pred = 2
                    for i in range(4,0,-1):
                        #use this exponential threshold thing, idk
                        thresh = np.exp2(-i)
                        if norm_rank < thresh:
                            pred = i
                            break
                else:
                    #impossible if not found in list
                    pred = 0
            else:
                #impossible if not found in list
                pred = 0
            if binary:
                pred = 1 if pred >= 3 else 0
            preds.append(pred)
    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help="file of triples to predict. Give a train file with corresponding dev/test files")
    parser.add_argument('--no-stats', dest='no_stats', action='store_const', const=True, help="flag to not do the stats part")
    args = parser.parse_args()

    #dict argmax function
    argmax = lambda d: sorted(d.items(), key=operator.itemgetter(1))[-1][0]

    #get part/whole counts from train
    whole_counts, whole_bin_counts = defaultdict(lambda: Counter()), defaultdict(lambda: Counter())
    part_counts, part_bin_counts = defaultdict(lambda: Counter()), defaultdict(lambda: Counter())
    jj_counts, jj_bin_counts = defaultdict(lambda: Counter()), defaultdict(lambda: Counter())
    label_counts, bin_label_counts = np.zeros(5), np.zeros(2)
    with open(args.file) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            whole, part, jj, label, bin_label = tuple(row)

            whole_counts[whole][int(label)] += 1
            whole_bin_counts[whole][int(bin_label)] += 1

            part_counts[part][int(label)] += 1
            part_bin_counts[part][int(bin_label)] += 1

            jj_counts[jj][int(label)] += 1
            jj_bin_counts[jj][int(bin_label)] += 1

            label_counts[int(label)] += 1
            bin_label_counts[int(bin_label)] += 1

    #use them to predict on dev/test
    most_common_dev, whole_dev, part_dev, jj_dev = [], [], [], []
    most_common_bin_dev, whole_bin_dev, part_bin_dev, jj_bin_dev = [], [], [], []
    gold_dev, gold_bin_dev = [], []
    with open(args.file.replace('train', 'dev')) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            whole, part, jj = tuple(row[:3])
            whole_dev.append(argmax(whole_counts[whole]) if whole in whole_counts else np.argmax(label_counts))
            whole_bin_dev.append(argmax(whole_bin_counts[whole]) if whole in whole_bin_counts else np.argmax(bin_label_counts))

            part_dev.append(argmax(part_counts[part]) if part in part_counts else np.argmax(label_counts))
            part_bin_dev.append(argmax(part_bin_counts[part]) if part in part_bin_counts else np.argmax(bin_label_counts))

            jj_dev.append(argmax(jj_counts[jj]) if jj in jj_counts else np.argmax(label_counts))
            jj_bin_dev.append(argmax(jj_bin_counts[jj]) if jj in jj_bin_counts else np.argmax(bin_label_counts))

            most_common_dev.append(np.argmax(label_counts))
            most_common_bin_dev.append(np.argmax(bin_label_counts))

            gold_dev.append(int(row[3]))
            gold_bin_dev.append(int(row[4]))

    #use them to predict on dev/test
    most_common_test, whole_test, part_test, jj_test = [], [], [], []
    most_common_bin_test, whole_bin_test, part_bin_test, jj_bin_test = [], [], [], []
    gold_test, gold_bin_test = [], []
    with open(args.file.replace('train', 'test')) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            whole, part = tuple(row[:2])
            whole_test.append(argmax(whole_counts[whole]) if whole in whole_counts else np.argmax(label_counts))
            whole_bin_test.append(argmax(whole_bin_counts[whole]) if whole in whole_bin_counts else np.argmax(bin_label_counts))

            part_test.append(argmax(part_counts[part]) if part in part_counts else np.argmax(label_counts))
            part_bin_test.append(argmax(part_bin_counts[part]) if part in part_bin_counts else np.argmax(bin_label_counts))

            jj_test.append(argmax(jj_counts[jj]) if jj in jj_counts else np.argmax(label_counts))
            jj_bin_test.append(argmax(jj_bin_counts[jj]) if jj in jj_bin_counts else np.argmax(bin_label_counts))

            most_common_test.append(np.argmax(label_counts))
            most_common_bin_test.append(np.argmax(bin_label_counts))

            gold_test.append(int(row[3]))
            gold_bin_test.append(int(row[4]))

    if not args.no_stats:
        stats_dev, stats_test = get_preds_by_ngram_stats(args.file)
        stats_bin_dev, stats_bin_test = get_preds_by_ngram_stats(args.file, binary=True)

    print("#" * 20 + " FULL LABEL RESULTS " + "#" * 20)
    print("acc, prec, rec, f1, MSE, spearman")
    print("MOST COMMON, dev: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_dev, most_common_dev),
                                                precision_score(gold_dev, most_common_dev, average='weighted'),
                                                recall_score(gold_dev, most_common_dev, average='weighted'),
                                                f1_score(gold_dev, most_common_dev, average='weighted'),
                                                mean_squared_error(gold_dev, most_common_dev),
                                                spearmanr(gold_dev, most_common_dev)[0]))
    print("MOST COMMON, test: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_test, most_common_test),
                                                precision_score(gold_test, most_common_test, average='weighted'),
                                                recall_score(gold_test, most_common_test, average='weighted'),
                                                f1_score(gold_test, most_common_test, average='weighted'),
                                                mean_squared_error(gold_test, most_common_test),
                                                spearmanr(gold_test, most_common_test)[0]))
    print("whole, dev: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_dev, whole_dev),
                                                precision_score(gold_dev, whole_dev, average='weighted'),
                                                recall_score(gold_dev, whole_dev, average='weighted'),
                                                f1_score(gold_dev, whole_dev, average='weighted'),
                                                mean_squared_error(gold_dev, whole_dev),
                                                spearmanr(gold_dev, whole_dev)[0]))
    print("whole, test: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_test, whole_test),
                                                precision_score(gold_test, whole_test, average='weighted'),
                                                recall_score(gold_test, whole_test, average='weighted'),
                                                f1_score(gold_test, whole_test, average='weighted'),
                                                mean_squared_error(gold_test, whole_test),
                                                spearmanr(gold_test, whole_test)[0]))
    print("part, dev: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_dev, part_dev),
                                                precision_score(gold_dev, part_dev, average='weighted'),
                                                recall_score(gold_dev, part_dev, average='weighted'),
                                                f1_score(gold_dev, part_dev, average='weighted'),
                                                mean_squared_error(gold_dev, part_dev),
                                                spearmanr(gold_dev, part_dev)[0]))
    print("part, test: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_test, part_test),
                                                precision_score(gold_test, part_test, average='weighted'),
                                                recall_score(gold_test, part_test, average='weighted'),
                                                f1_score(gold_test, part_test, average='weighted'),
                                                mean_squared_error(gold_test, part_test),
                                                spearmanr(gold_test, part_test)[0]))
    print("jj, dev: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_dev, jj_dev),
                                                precision_score(gold_dev, jj_dev, average='weighted'),
                                                recall_score(gold_dev, jj_dev, average='weighted'),
                                                f1_score(gold_dev, jj_dev, average='weighted'),
                                                mean_squared_error(gold_dev, jj_dev),
                                                spearmanr(gold_dev, jj_dev)[0]))
    print("jj, test: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_test, jj_test),
                                                precision_score(gold_test, jj_test, average='weighted'),
                                                recall_score(gold_test, jj_test, average='weighted'),
                                                f1_score(gold_test, jj_test, average='weighted'),
                                                mean_squared_error(gold_test, jj_test),
                                                spearmanr(gold_test, jj_test)[0]))
    if not args.no_stats:
        print("stats, dev: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_dev, stats_dev),
                                                    precision_score(gold_dev, stats_dev, average='weighted'),
                                                    recall_score(gold_dev, stats_dev, average='weighted'),
                                                    f1_score(gold_dev, stats_dev, average='weighted'),
                                                    mean_squared_error(gold_dev, stats_dev),
                                                    spearmanr(gold_dev, stats_dev)[0]))
        print("stats, test: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_test, stats_test),
                                                    precision_score(gold_test, stats_test, average='weighted'),
                                                    recall_score(gold_test, stats_test, average='weighted'),
                                                    f1_score(gold_test, stats_test, average='weighted'),
                                                    mean_squared_error(gold_test, stats_test),
                                                    spearmanr(gold_test, stats_test)[0]))

    print("whole dev confusion matrix")
    whole_conmat = confusion_matrix(gold_dev, whole_dev)
    print(whole_conmat)
    labels = ['impossible', 'unlikely', 'unrelated', 'probably', 'guaranteed']
    fig = plt.figure(1)
    fig.set_size_inches(14,4)
    ticks = np.arange(5)
    ax = plt.subplot(131)

    wc = whole_conmat.copy()
    for i in range(5):
        wc[i][i] = 0
    plt.imshow(wc, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    thresh = wc.max() / 2
    for i,j in itertools.product(range(5), range(5)):
        plt.text(j,i,whole_conmat[i][j],horizontalalignment='center',color='white' if wc[i][j] > thresh else 'black')

    #df_cmw = pd.DataFrame(whole_conmat, index=labels, columns=labels)
    #sns.heatmap(df_cmw, annot=True, fmt='g', annot_kws={'fontsize': 'xx-large'})
    ax.set_title('Most common per whole')
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')

    print("part dev confusion matrix")
    part_conmat = confusion_matrix(gold_dev, part_dev)
    print(part_conmat)
    ax = plt.subplot(132)

    pc = part_conmat.copy()
    for i in range(5):
        pc[i][i] = 0
    plt.imshow(pc, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    thresh = pc.max() / 2
    for i,j in itertools.product(range(5), range(5)):
        plt.text(j,i,part_conmat[i][j],horizontalalignment='center',color='white' if pc[i][j] > thresh else 'black')

    #df_cmp = pd.DataFrame(part_conmat, index=labels, columns=labels)
    #sns.heatmap(df_cmp, annot=True, fmt='g', annot_kws={'fontsize': 'xx-large'})
    ax.set_title('Most common per part')
    ax.set_xlabel('Predicted class')

    print("jj dev confusion matrix")
    jj_conmat = confusion_matrix(gold_dev, jj_dev)
    print(jj_conmat)
    ax = plt.subplot(133)

    jc = jj_conmat.copy()
    for i in range(5):
        jc[i][i] = 0
    plt.imshow(jc, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    thresh = jc.max() / 2
    for i,j in itertools.product(range(5), range(5)):
        plt.text(j,i,jj_conmat[i][j],horizontalalignment='center',color='white' if jc[i][j] > thresh else 'black')

    #df_cmj = pd.DataFrame(jj_conmat, index=labels, columns=labels)
    #sns.heatmap(df_cmj, annot=True, fmt='g', annot_kws={'fontsize': 'xx-large'})
    ax.set_title('Most common per adjective')
    ax.set_xlabel('Predicted class')
    plt.tight_layout()
    plt.show()

    print()
    print("#" * 20 + " BINARY LABEL RESULTS " + "#" * 20)
    print("acc, prec, rec, f1, spearman")
    print("MOST COMMON, dev: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_bin_dev, most_common_bin_dev),
                                                precision_score(gold_bin_dev, most_common_bin_dev, average='weighted'),
                                                recall_score(gold_bin_dev, most_common_bin_dev, average='weighted'),
                                                f1_score(gold_bin_dev, most_common_bin_dev, average='weighted'),
                                                mean_squared_error(gold_bin_dev, most_common_bin_dev),
                                                spearmanr(gold_bin_dev, most_common_bin_dev)[0]))
    print("MOST COMMON, test: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_bin_test, most_common_bin_test),
                                                precision_score(gold_bin_test, most_common_bin_test, average='weighted'),
                                                recall_score(gold_bin_test, most_common_bin_test, average='weighted'),
                                                f1_score(gold_bin_test, most_common_bin_test, average='weighted'),
                                                mean_squared_error(gold_bin_test, most_common_bin_test),
                                                spearmanr(gold_bin_test, most_common_bin_test)[0]))
    print("whole, dev: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_bin_dev, whole_bin_dev),
                                                precision_score(gold_bin_dev, whole_bin_dev, average='weighted'),
                                                recall_score(gold_bin_dev, whole_bin_dev, average='weighted'),
                                                f1_score(gold_bin_dev, whole_bin_dev, average='weighted'),
                                                mean_squared_error(gold_bin_dev, whole_bin_dev),
                                                spearmanr(gold_bin_dev, whole_bin_dev)[0]))
    print("whole, test: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_bin_test, whole_bin_test),
                                                precision_score(gold_bin_test, whole_bin_test, average='weighted'),
                                                recall_score(gold_bin_test, whole_bin_test, average='weighted'),
                                                f1_score(gold_bin_test, whole_bin_test, average='weighted'),
                                                mean_squared_error(gold_bin_test, whole_bin_test),
                                                spearmanr(gold_bin_test, whole_bin_test)[0]))
    print("part, dev: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_bin_dev, part_bin_dev),
                                                precision_score(gold_bin_dev, part_bin_dev, average='weighted'),
                                                recall_score(gold_bin_dev, part_bin_dev, average='weighted'),
                                                f1_score(gold_bin_dev, part_bin_dev, average='weighted'),
                                                mean_squared_error(gold_bin_dev, part_bin_dev),
                                                spearmanr(gold_bin_dev, part_bin_dev)[0]))
    print("part, test: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_bin_test, part_bin_test),
                                                precision_score(gold_bin_test, part_bin_test, average='weighted'),
                                                recall_score(gold_bin_test, part_bin_test, average='weighted'),
                                                f1_score(gold_bin_test, part_bin_test, average='weighted'),
                                                mean_squared_error(gold_bin_test, part_bin_test),
                                                spearmanr(gold_bin_test, part_bin_test)[0]))
    print("jj, dev: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_bin_dev, jj_bin_dev),
                                                precision_score(gold_bin_dev, jj_bin_dev, average='weighted'),
                                                recall_score(gold_bin_dev, jj_bin_dev, average='weighted'),
                                                f1_score(gold_bin_dev, jj_bin_dev, average='weighted'),
                                                mean_squared_error(gold_bin_dev, jj_bin_dev),
                                                spearmanr(gold_bin_dev, jj_bin_dev)[0]))
    print("jj, test: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_bin_test, jj_bin_test),
                                                precision_score(gold_bin_test, jj_bin_test, average='weighted'),
                                                recall_score(gold_bin_test, jj_bin_test, average='weighted'),
                                                f1_score(gold_bin_test, jj_bin_test, average='weighted'),
                                                mean_squared_error(gold_bin_test, jj_bin_test),
                                                spearmanr(gold_bin_test, jj_bin_test)[0]))
    if not args.no_stats:
        print("stats, dev: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_bin_dev, stats_bin_dev),
                                                    precision_score(gold_bin_dev, stats_bin_dev, average='weighted'),
                                                    recall_score(gold_bin_dev, stats_bin_dev, average='weighted'),
                                                    f1_score(gold_bin_dev, stats_bin_dev, average='weighted'),
                                                    mean_squared_error(gold_bin_dev, stats_bin_dev),
                                                    spearmanr(gold_bin_dev, stats_bin_dev)[0]))
        print("stats, test: %.3f & %.3f & %.3f & %.3f & %01.2f & %.3f" % (accuracy_score(gold_bin_test, stats_bin_test),
                                                    precision_score(gold_bin_test, stats_bin_test, average='weighted'),
                                                    recall_score(gold_bin_test, stats_bin_test, average='weighted'),
                                                    f1_score(gold_bin_test, stats_bin_test, average='weighted'),
                                                    mean_squared_error(gold_bin_test, stats_bin_test),
                                                    spearmanr(gold_bin_test, stats_bin_test)[0]))

