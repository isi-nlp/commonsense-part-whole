import argparse, csv
from collections import defaultdict
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str, help="path to file to get scores for")
    parser.add_argument("num_assignments", type=int, help="number of turkers on each HIT")
    parser.add_argument("--two-class", dest="two_class", const=True, action="store_const", help="get two-class metrics instead of full-range")
    args = parser.parse_args()

    str2score = {'guaranteed': 4, 'probably': 3, 'unrelated': 2, 'unlikely': 1, 'impossible': 0}
    str2score2 = {'guaranteed': 1, 'probably': 1, 'unrelated': 0, 'unlikely': 0, 'impossible': 0}

    gold = {tuple(row[:3]):row[3] for row in csv.reader(open('gold_annotations.csv'))}
    worker_responses = defaultdict(dict) #worker id : {triple: annotation}
    with open(args.results_file) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            triple = tuple(row[3:6])
            for (res, worker) in zip(row[6::2], row[7::2]):
                worker_responses[worker][triple] = res

    #worker : agreement
    k_agreements = {}
    k_linear_agreements = {}
    k_quadratic_agreements = {}
    k_two_class_agreements = {}
    rho_agreements = {}
    for worker, responses in worker_responses.items():
        x, y = [], []
        x2, y2 = [], []
        for triple, gold_response in gold.items():
            response = responses[triple]
            if response is not None and 'nonsense' not in response:
                x.append(str2score[gold_response])
                y.append(str2score[response])
                x2.append(str2score2[gold_response])
                y2.append(str2score2[response])
        if len(x) > 0 and len(y) > 0:
            k_agreements[worker] = cohen_kappa_score(x, y)
            k_linear_agreements[worker] = cohen_kappa_score(x, y, weights='linear')
            k_quadratic_agreements[worker] = cohen_kappa_score(x, y, weights='quadratic')
            k_two_class_agreements[worker] = cohen_kappa_score(x2, y2)

            rho, pval = spearmanr(x, y)
            rho_agreements[worker] = rho

    with open('qualified_pool.csv', 'w') as of:
        w = csv.writer(of)
        w.writerow(['worker_id','kappa','kappa_linear','kappa_quadratic','kappa_two_class','spearman'])
        for worker in k_agreements.keys():
            w.writerow([worker, k_agreements[worker], k_linear_agreements[worker], k_quadratic_agreements[worker], k_two_class_agreements[worker], rho_agreements[worker]])
