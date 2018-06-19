import argparse, csv, itertools
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_id", type=str, help="batch id to get kappa score for")
    parser.add_argument("--weights", required=False, choices=['linear', 'quadratic', 'None'], default='linear', help="weighting to use for cohen's kappa metric (default: linear)")
    args = parser.parse_args()
    args.weights = None if args.weights == 'None' else args.weights

    str2score = {'guaranteed': 4, 'probably': 3, 'unrelated': 2, 'unlikely': 1, 'impossible': 0}
    with open("hit_results/batch_%s.csv" % args.batch_id) as f:
        r = csv.reader(f)
        #header
        next(r)
        res1, res2, res3 = [], [], []
        for row in r:
            responses = row[6:-1:2]
            if len(responses) > 1:
                res1.append(responses[0])
                res2.append(responses[1])
                if len(responses) > 2:
                    res3.append(responses[2])
                else:
                    res3.append(None)
        ks = []
        rs = []
        for (x, y) in itertools.combinations([res1, res2, res3], 2):
            x2, y2 = [], []
            for x_i, y_i in zip(x, y):
                if x_i is not None and y_i is not None and 'nonsense' not in x_i and 'nonsense' not in y_i:
                    x2.append(str2score[x_i])
                    y2.append(str2score[y_i])
            k = cohen_kappa_score(x2, y2, weights=args.weights)
            ks.append(k)
            rho, pval = spearmanr(x2, y2)
            rs.append(rho)
        print("#" * 10 + " COHEN'S KAPPA  " + "#" * 10)
        print(ks)
        print("AVERAGE: %f +/- %f" % (np.mean(ks), np.std(ks)))
        print()
        print("#" * 10 + " SPEARMAN'S RHO " + "#" * 10)
        print(rs)
        print("AVERAGE: %f +/- %f" % (np.mean(rs), np.std(rs)))
        

