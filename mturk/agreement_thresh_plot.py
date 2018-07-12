import argparse, csv, itertools
from collections import Counter, defaultdict, namedtuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
from tqdm import tqdm

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

    annotations = defaultdict(dict)
    all_trips = set()
    with open(args.results_file) as f:
        r = csv.reader(f)
        #header
        next(r)
        response_arrs = [[] for i in range(args.num_assignments)]
        for row in r:
            responses = row[6::2]
            worker_ids = row[7::2]
            trip = '_'.join(' '.join(row[3:6]).split())
            if len(responses) > 1:
                for i,(res,worker) in enumerate(zip(responses, worker_ids)):
                    if i > args.num_assignments - 1:
                        break
                    annotations[worker][trip] = res
                    all_trips.add(trip)

    pair2k = {}
    pair2shared = {}
    ks = []
    for w1, w2 in tqdm(itertools.combinations(annotations.keys(), 2)):
        shared = list(set(annotations[w1].keys()).intersection(annotations[w2].keys()))
        valid_shared = []
        x, y = [annotations[w1][k] for k in shared], [annotations[w2][k] for k in shared]
        x2, y2 = [], []
        for trip, x_i, y_i in zip(shared, x, y):
            if x_i is not None and y_i is not None:
                if 'nonsense' not in x_i or (args.pjj_equals_impossible and x_i == 'pjj-nonsense'):
                    if 'nonsense' not in y_i or (args.pjj_equals_impossible and y_i == 'pjj-nonsense'):
                        x2.append(str2score[x_i])
                        y2.append(str2score[y_i])
                        valid_shared.append(trip)
        k = cohen_kappa_score(x2, y2, weights=args.weights)
        if not np.isnan(k) and len(valid_shared) > 1:
            ks.append(k)
            pair2shared[(w1, w2)] = valid_shared
            pair2k[(w1, w2)] = k

    # Now we have K scores between each worker that had any overlap
    # IDK how they JHU actually made their plot / did their analysis
    # setting a threshold of 0.5 and keeping only examples where all worker pairs had that agreement is v harsh
    # maybe they keep any example where at least one worker pair has that level of agreement?
    # my plot's shape looks very different if I do that though

    threshes = np.arange(0,1,0.01)
    data_szs = []
    for thresh in tqdm(threshes):
        badpairs = [pair for pair,k in pair2k.items() if k < thresh]
        goodpairs = [pair for pair,k in pair2k.items() if k >= thresh]
        trip_counter = Counter()
        for pair in goodpairs:
            for trip in pair2shared[pair]:
                trip_counter[trip] += 1
        goodtrips = [trip for trip,count in trip_counter.items() if count > 0]
        data_szs.append(len(goodtrips))

    plt.plot(threshes, data_szs)
    plt.xlim(1,0)
    plt.show()

    #for trip, ann1, ann2 in zip(pair2shared[pair], [annotations[pair[0]][k] for k in pair2shared[pair]], [annotations[pair[1]][k] for k in pair2shared[pair]]):
    #    print(trip, ann1, ann2)
