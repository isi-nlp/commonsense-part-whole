"""
    Calculate average human accuracy on given dataset containing all annotations and ground truth label
"""
import argparse
import csv
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="path to file to get human acc for")
    args = parser.parse_args()
    preds = [[] for _ in range(5)]
    golds = [[] for _ in range(5)]
    with open(args.file) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            for ix,response in enumerate(row[5:]):
                preds[ix].append(int(response))
                golds[ix].append(int(row[3]))
    accs, bin_accs = [], []
    precs, bin_precs = [], []
    recs, bin_recs = [], []
    f1s, bin_f1s = [], []
    mses, bin_mses = [], []
    rhos, bin_rhos = [], []
    ord2bin = [0, 0, 0, 1, 1]
    for gold, pred in zip(golds, preds):
        bin_gold = [ord2bin[g] for g in gold]
        bin_pred = [ord2bin[p] for p in pred]
        accs.append(accuracy_score(gold, pred))
        bin_accs.append(accuracy_score(bin_gold, bin_pred))
        precs.append(precision_score(gold, pred, average='weighted'))
        bin_precs.append(precision_score(bin_gold, bin_pred))
        recs.append(recall_score(gold, pred, average='weighted'))
        bin_recs.append(recall_score(bin_gold, bin_pred))
        f1s.append(f1_score(gold, pred, average='weighted'))
        bin_f1s.append(f1_score(bin_gold, bin_pred))
        mses.append(mean_squared_error(gold, pred))
        bin_mses.append(mean_squared_error(bin_gold, bin_pred))
        rhos.append(spearmanr(gold, pred)[0])
        bin_rhos.append(spearmanr(bin_gold, bin_pred)[0])
    #weighted averages
    tot = sum([len(p) for p in preds])
    acc = sum([len(p)/tot*a for p,a in zip(preds, accs)])
    bin_acc = sum([len(p)/tot*a for p,a in zip(preds, bin_accs)])
    prec = sum([len(p)/tot*a for p,a in zip(preds, precs)])
    bin_prec = sum([len(p)/tot*a for p,a in zip(preds, bin_precs)])
    rec = sum([len(p)/tot*a for p,a in zip(preds, recs)])
    bin_rec = sum([len(p)/tot*a for p,a in zip(preds, bin_recs)])
    f1 = sum([len(p)/tot*a for p,a in zip(preds, f1s)])
    bin_f1 = sum([len(p)/tot*a for p,a in zip(preds, bin_f1s)])
    mse = sum([len(p)/tot*a for p,a in zip(preds, mses)])
    bin_mse = sum([len(p)/tot*a for p,a in zip(preds, bin_mses)])
    rho = sum([len(p)/tot*a for p,a in zip(preds, rhos)])
    bin_rho = sum([len(p)/tot*a for p,a in zip(preds, bin_rhos)])
    print("ORDINAL")
    print(f"ACC: {acc}")
    print(f"PREC: {prec}")
    print(f"REC: {rec}")
    print(f"F1: {f1}")
    print(f"MSE: {mse}")
    print(f"RHO: {rho}")

    print("BINARY")
    print(f"ACC: {bin_acc}")
    print(f"PREC: {bin_prec}")
    print(f"REC: {bin_rec}")
    print(f"F1: {bin_f1}")
    print(f"MSE: {bin_mse}")
    print(f"RHO: {bin_rho}")
