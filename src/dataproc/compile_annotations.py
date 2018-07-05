"""
    Read a file of AMT output (all_just_results.csv), take the median response as ground truth
    Split into train/dev/test with given split
"""
import argparse, csv, random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to AMT data')
    parser.add_argument('outfile', type=str, help='path to write annotated data. base name: will append _train, _dev, _test')
    parser.add_argument('split', type=str, help='train/test/dev split, given slash delimited like 70/10/20')
    parser.add_argument('--pjj-equals-impossible', const=True, action='store_const', help='flag to let pjj-nonsense be equal to impossible')
    args = parser.parse_args()

    split = [int(s)/100 for s in args.split.split('/')]
    print(split)
    train_sz, dev_sz, test_sz = split

    str2score = {'guaranteed': 4, 'probably': 3, 'unrelated': 2, 'unlikely': 1, 'impossible': 0}
    str2score2 = {'guaranteed': 1, 'probably': 1, 'unrelated': 0, 'unlikely': 0, 'impossible': 0}
    if args.pjj_equals_impossible:
        str2score['pjj-nonsense'] = 0
        str2score2['pjj-nonsense'] = 0

    examples = []
    with open(args.file) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            triple = tuple(row[:3])

            trash = False
            responses, bin_responses = [], []
            for rs in row[3:]:
                if 'nonsense' in rs and not (args.pjj_equals_impossible and rs == 'pjj-nonsense'):
                    trash = True
            if trash:
                continue

            responses = [str2score[rs] for rs in row[3:]]
            bin_responses = [str2score2[rs] for rs in row[3:]]

            if len(responses) % 2 == 0:
                #if we have an even number of responses for some reason, randomly delete one
                idx = random.randrange(0,len(responses))
                del(responses[idx])
            if len(bin_responses) % 2 == 0:
                #if we have an even number of bin_responses for some reason, randomly delete one
                idx = random.randrange(0,len(bin_responses))
                del(bin_responses[idx])

            label = int(np.median(responses))
            bin_label = int(np.median(bin_responses))
            examples.append([*triple, label, bin_label])

    trf = open(args.outfile.replace('.csv', '_train.csv'), 'w')
    dvf = open(args.outfile.replace('.csv', '_dev.csv'), 'w')
    tef = open(args.outfile.replace('.csv', '_test.csv'), 'w')

    trw = csv.writer(trf)
    trw.writerow(['whole', 'part', 'jj', 'label', 'bin_label'])
    dvw = csv.writer(dvf)
    dvw.writerow(['whole', 'part', 'jj', 'label', 'bin_label'])
    tew = csv.writer(tef)
    tew.writerow(['whole', 'part', 'jj', 'label', 'bin_label'])
    
    random.shuffle(examples)
    for i,example in enumerate(examples):
        rank = i / len(examples)
        if rank < train_sz:
            trw.writerow(example)
        elif rank < train_sz + dev_sz:
            dvw.writerow(example)
        else:
            tew.writerow(example)

    trf.close()
    dvf.close()
    tef.close()
