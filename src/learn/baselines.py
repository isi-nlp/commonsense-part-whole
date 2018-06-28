"""
    Given a file of triples, predicts the labels from:
    - most common overall class
    - most common class for given part (default to most common overall if unseen)
    - most common class for given whole (default to most common overall if unseen)
    - statistics from Google n-grams on the frequency of JJ amod PART
"""
import argparse, csv, operator
from collections import Counter, defaultdict, OrderedDict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    args = parser.parse_args()

    #dict argmax function
    argmax = lambda d: sorted(d.items(), key=operator.itemgetter(1))[-1][0]

    #get part/whole counts from train
    whole_counts, whole_bin_counts = defaultdict(lambda: Counter()), defaultdict(lambda: Counter())
    part_counts, part_bin_counts = defaultdict(lambda: Counter()), defaultdict(lambda: Counter())
    label_counts, bin_label_counts = np.zeros(5), np.zeros(2)
    with open(args.file) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            whole, part, _, label, bin_label = tuple(row)

            whole_counts[whole][int(label)] += 1
            whole_bin_counts[whole][int(bin_label)] += 1

            part_counts[part][int(label)] += 1
            part_bin_counts[part][int(bin_label)] += 1

            label_counts[int(label)] += 1
            bin_label_counts[int(bin_label)] += 1

    #use them to predict on dev/test
    most_common_dev, whole_dev, part_dev = [], [], []
    most_common_bin_dev, whole_bin_dev, part_bin_dev = [], [], []
    gold_dev, gold_bin_dev = [], []
    with open(args.file.replace('train', 'dev')) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            whole, part = tuple(row[:2])
            whole_dev.append(argmax(whole_counts[whole]) if whole in whole_counts else np.argmax(label_counts))
            whole_bin_dev.append(argmax(whole_bin_counts[whole]) if whole in whole_bin_counts else np.argmax(bin_label_counts))

            part_dev.append(argmax(part_counts[part]) if part in part_counts else np.argmax(label_counts))
            part_bin_dev.append(argmax(part_bin_counts[part]) if part in part_bin_counts else np.argmax(bin_label_counts))

            most_common_dev.append(np.argmax(label_counts))
            most_common_bin_dev.append(np.argmax(bin_label_counts))

            gold_dev.append(int(row[3]))
            gold_bin_dev.append(int(row[4]))

    #use them to predict on dev/test
    most_common_test, whole_test, part_test = [], [], []
    most_common_bin_test, whole_bin_test, part_bin_test = [], [], []
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

            most_common_test.append(np.argmax(label_counts))
            most_common_bin_test.append(np.argmax(bin_label_counts))

            gold_test.append(int(row[3]))
            gold_bin_test.append(int(row[4]))

    stats_dev, stats_test = get_preds_by_ngram_stats(args.file)
    stats_bin_dev, stats_bin_test = get_preds_by_ngram_stats(args.file, binary=True)

    print("#" * 20 + " FULL LABEL RESULTS " + "#" * 20)
    print("acc, prec, rec, f1")
    print("MOST COMMON, dev: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_dev, most_common_dev),
                                                precision_score(gold_dev, most_common_dev, average='weighted'),
                                                recall_score(gold_dev, most_common_dev, average='weighted'),
                                                f1_score(gold_dev, most_common_dev, average='weighted')))
    print("MOST COMMON, test: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_test, most_common_test),
                                                precision_score(gold_test, most_common_test, average='weighted'),
                                                recall_score(gold_test, most_common_test, average='weighted'),
                                                f1_score(gold_test, most_common_test, average='weighted')))
    print("whole, dev: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_dev, whole_dev),
                                                precision_score(gold_dev, whole_dev, average='weighted'),
                                                recall_score(gold_dev, whole_dev, average='weighted'),
                                                f1_score(gold_dev, whole_dev, average='weighted')))
    print("whole, test: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_test, whole_test),
                                                precision_score(gold_test, whole_test, average='weighted'),
                                                recall_score(gold_test, whole_test, average='weighted'),
                                                f1_score(gold_test, whole_test, average='weighted')))
    print("part, dev: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_dev, part_dev),
                                                precision_score(gold_dev, part_dev, average='weighted'),
                                                recall_score(gold_dev, part_dev, average='weighted'),
                                                f1_score(gold_dev, part_dev, average='weighted')))
    print("part, test: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_test, part_test),
                                                precision_score(gold_test, part_test, average='weighted'),
                                                recall_score(gold_test, part_test, average='weighted'),
                                                f1_score(gold_test, part_test, average='weighted')))
    print("stats, dev: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_dev, stats_dev),
                                                precision_score(gold_dev, stats_dev, average='weighted'),
                                                recall_score(gold_dev, stats_dev, average='weighted'),
                                                f1_score(gold_dev, stats_dev, average='weighted')))
    print("stats, test: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_test, stats_test),
                                                precision_score(gold_test, stats_test, average='weighted'),
                                                recall_score(gold_test, stats_test, average='weighted'),
                                                f1_score(gold_test, stats_test, average='weighted')))

    print()
    print("#" * 20 + " BINARY LABEL RESULTS " + "#" * 20)
    print("acc, prec, rec, f1")
    print("MOST COMMON, dev: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_bin_dev, most_common_bin_dev),
                                                precision_score(gold_bin_dev, most_common_bin_dev, average='weighted'),
                                                recall_score(gold_bin_dev, most_common_bin_dev, average='weighted'),
                                                f1_score(gold_bin_dev, most_common_bin_dev, average='weighted')))
    print("MOST COMMON, test: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_bin_test, most_common_bin_test),
                                                precision_score(gold_bin_test, most_common_bin_test, average='weighted'),
                                                recall_score(gold_bin_test, most_common_bin_test, average='weighted'),
                                                f1_score(gold_bin_test, most_common_bin_test, average='weighted')))
    print("whole, dev: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_bin_dev, whole_bin_dev),
                                                precision_score(gold_bin_dev, whole_bin_dev, average='weighted'),
                                                recall_score(gold_bin_dev, whole_bin_dev, average='weighted'),
                                                f1_score(gold_bin_dev, whole_bin_dev, average='weighted')))
    print("whole, test: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_bin_test, whole_bin_test),
                                                precision_score(gold_bin_test, whole_bin_test, average='weighted'),
                                                recall_score(gold_bin_test, whole_bin_test, average='weighted'),
                                                f1_score(gold_bin_test, whole_bin_test, average='weighted')))
    print("part, dev: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_bin_dev, part_bin_dev),
                                                precision_score(gold_bin_dev, part_bin_dev, average='weighted'),
                                                recall_score(gold_bin_dev, part_bin_dev, average='weighted'),
                                                f1_score(gold_bin_dev, part_bin_dev, average='weighted')))
    print("part, test: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_bin_test, part_bin_test),
                                                precision_score(gold_bin_test, part_bin_test, average='weighted'),
                                                recall_score(gold_bin_test, part_bin_test, average='weighted'),
                                                f1_score(gold_bin_test, part_bin_test, average='weighted')))
    print("stats, dev: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_bin_dev, stats_bin_dev),
                                                precision_score(gold_bin_dev, stats_bin_dev, average='weighted'),
                                                recall_score(gold_bin_dev, stats_bin_dev, average='weighted'),
                                                f1_score(gold_bin_dev, stats_bin_dev, average='weighted')))
    print("stats, test: %.3f, %.3f, %.3f, %.3f" % (accuracy_score(gold_bin_test, stats_bin_test),
                                                precision_score(gold_bin_test, stats_bin_test, average='weighted'),
                                                recall_score(gold_bin_test, stats_bin_test, average='weighted'),
                                                f1_score(gold_bin_test, stats_bin_test, average='weighted')))
    import pdb; pdb.set_trace()
