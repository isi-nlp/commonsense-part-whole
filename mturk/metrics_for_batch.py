import argparse, csv, itertools
from collections import defaultdict
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str, help="path to file to get scores for")
    parser.add_argument("num_assignments", type=int, help="number of turkers on each HIT")
    parser.add_argument("--weights", required=False, choices=['linear', 'quadratic', 'None'], default='linear', help="weighting to use for cohen's kappa metric (default: linear)")
    parser.add_argument("--two-class", dest="two_class", const=True, action="store_const", help="get two-class metrics instead of full-range")
    parser.add_argument("--pjj-equals-impossible", dest="pjj_equals_impossible", const=True, action="store_const", help="flag to set pjj-nonsense equal to impossible")
    args = parser.parse_args()
    args.weights = None if args.weights == 'None' else args.weights

    if args.two_class:
        str2score = {'guaranteed': 1, 'probably': 1, 'unrelated': 0, 'unlikely': 0, 'impossible': 0}
    else:
        str2score = {'guaranteed': 4, 'probably': 3, 'unrelated': 2, 'unlikely': 1, 'impossible': 0}
    if args.pjj_equals_impossible:
        str2score['pjj-nonsense'] = 0

    with open(args.results_file) as f:
        r = csv.reader(f)
        #header
        next(r)
        response_arrs = [[] for i in range(args.num_assignments)]
        for row in r:
            responses = row[6::2]
            if len(responses) > 1:
                for i,res in enumerate(responses):
                    response_arrs[i].append(res)
                    if i >= args.num_assignments - 1:
                        break
                #add Nones for any missing
                for j in range(i+1, args.num_assignments):
                    response_arrs[j].append(None)


    pair2k = {}
    pair2shared = {}

    ks = []
    rs = []
    for (i, j) in itertools.combinations(range(args.num_assignments), 2):
        x, y = response_arrs[i], response_arrs[j]
        x2, y2 = [], []
        for x_i, y_i in zip(x, y):
            if x_i is not None and y_i is not None:
                if 'nonsense' not in x_i or (args.pjj_equals_impossible and x_i == 'pjj-nonsense'):
                    if 'nonsense' not in y_i or (args.pjj_equals_impossible and y_i == 'pjj-nonsense'):
                        x2.append(str2score[x_i])
                        y2.append(str2score[y_i])
        k = cohen_kappa_score(x2, y2, weights=args.weights)
        ks.append(k)
        rho, pval = spearmanr(x2, y2)
        rs.append(rho)
    print("#" * 10 + " COHEN'S KAPPA  " + "#" * 10)
    print(ks)
    print("AVERAGE: %f +/- %f" % (np.nanmean(ks), np.nanstd(ks)))
    print()
    print("#" * 10 + " SPEARMAN'S RHO " + "#" * 10)
    print(rs)
    print("AVERAGE: %f +/- %f" % (np.nanmean(rs), np.nanstd(rs)))
        

