import argparse, csv
from collections import defaultdict
import boto3
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

MTURK_URL = 'https://mturk-requester.us-east-1.amazonaws.com'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str, help="path to file to get scores for")
    parser.add_argument("num_assignments", type=int, help="number of turkers on each HIT")
    parser.add_argument("metric", choices=['kappa', 'kappa_linear', 'kappa_quadratic', 'kappa_two_class', 'spearman'], help="which metric criterion to use")
    parser.add_argument("threshold", type=float, help="workers must be above this threshold of agreement to qualify")
    parser.add_argument("--pjj-equals-impossible", dest='pjj_equals_impossible', const=True, action='store_const', 
                        help="flag to consider 'pjj-nonsense' response as equivalent to 'impossible'")
    parser.add_argument("--nonsense-equals-impossible", dest='nonsense_equals_impossible', const=True, action='store_const', 
                        help="flag to consider any nonsense response as equivalent to 'impossible'")
    parser.add_argument("--live", const=True, action='store_const', help="flag to actually run this and give workers a qualification")
    parser.add_argument("--create", const=True, action='store_const', help="flag to actually run this and give workers a NEW qualification")
    args = parser.parse_args()

    ######### AWS STUFF #########
    with open('/home/jamesm/.aws/credentials2.csv') as f:
        r = csv.reader(f)
        next(r)
        creds = next(r)
        iam_access = creds[2]
        iam_secret = creds[3]

    mturk = boto3.client('mturk',
                         aws_access_key_id = iam_access,
                         aws_secret_access_key = iam_secret,
                         region_name='us-east-1',
                         endpoint_url = MTURK_URL
                         )
    #############################

    str2score = {'guaranteed': 4, 'probably': 3, 'unrelated': 2, 'unlikely': 1, 'impossible': 0}
    str2score2 = {'guaranteed': 1, 'probably': 1, 'unrelated': 0, 'unlikely': 0, 'impossible': 0}
    if args.pjj_equals_impossible or args.nonsense_equals_impossible:
        str2score['pjj-nonsense'] = 0
        str2score2['pjj-nonsense'] = 0
        if args.nonsense_equals_impossible:
            str2score['wjj-nonsense'] = 0
            str2score2['wjj-nonsense'] = 0
            str2score['pw-nonsense'] = 0
            str2score2['pw-nonsense'] = 0
            str2score['nonsense'] = 0
            str2score2['nonsense'] = 0

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

    agreements = defaultdict(dict) #metric: (worker: agreement)
    for worker, responses in worker_responses.items():
        x, y = [], []
        x2, y2 = [], []
        for triple, gold_response in gold.items():
            response = responses[triple]
            if response is not None:
                if response == 'word-nonsense': continue
                if 'nonsense' not in response or (args.pjj_equals_impossible and response == 'pjj-nonsense') or args.nonsense_equals_impossible:
                    x.append(str2score[gold_response])
                    y.append(str2score[response])
                    x2.append(str2score2[gold_response])
                    y2.append(str2score2[response])
        if len(x) > 6 and len(y) > 6:
            agreements['kappa'][worker] = cohen_kappa_score(x, y)
            agreements['kappa_linear'][worker] = cohen_kappa_score(x, y, weights='linear')
            agreements['kappa_quadratic'][worker] = cohen_kappa_score(x, y, weights='quadratic')
            agreements['kappa_two_class'][worker] = cohen_kappa_score(x2, y2)

            rho, pval = spearmanr(x, y)
            agreements['spearman'][worker] = rho

    if args.live:
        cont = input('About to assign qualifications. You sure? (say yes if so) > ')
        if cont == 'yes':
            if args.create:
                print("creating qualification")
                qual = mturk.create_qualification_type(
                                     Name='commonsense-visual-qualifier',
                                     QualificationTypeStatus='Active',
                                     Description='Invited to participate in the full common sense visual reasoning study'
                                     )
                qual = qual['QualificationType']
                qual_id = qual['QualificationTypeId']
                with open('qualification_id', 'w') as of:
                    w = csv.writer(of)
                    w.writerow(['id', 'name', 'description'])
                    w.writerow([str(qual_id), qual['Name'], qual['Description']])
            else:
                with open('qualification_id') as f:
                    r = csv.reader(f)
                    #header
                    next(r)
                    qual_id, name, desc = tuple(next(r))

    if args.live and cont == 'yes':
        print("assigning qualifications")
        prev_qualified = []
        response = mturk.list_workers_with_qualification_type(QualificationTypeId='3J4VHEWGB82EEK2P3BTS7BQAJN38VS', MaxResults=100)
        while response['NumResults'] > 0:
            tok = response['NextToken']
            prev_qualified.extend([qual['WorkerId'] for qual in response['Qualifications']])
            response = mturk.list_workers_with_qualification_type(QualificationTypeId='3J4VHEWGB82EEK2P3BTS7BQAJN38VS', MaxResults=100, NextToken=tok)

    qualified_workers = []
    for worker in agreements['kappa'].keys():
        if agreements[args.metric][worker] > args.threshold:
            qualified_workers.append(worker)
            if args.live:
                print(worker)
                if worker not in prev_qualified:
                    mturk.associate_qualification_with_worker(QualificationTypeId=qual_id, WorkerId=worker)
    print("Number of qualified workers: %d/%d" % (len(qualified_workers), len(agreements['kappa'].keys())))
