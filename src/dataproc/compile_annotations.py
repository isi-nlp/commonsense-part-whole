"""
    Read a file of AMT output (i.e., all_just_results.csv)
    Throw a result out if there's a nonsense annotation
    Take the median response as ground truth
    Split into train/dev/test with given split percentages
"""
import argparse, csv, random
import numpy as np

def write_splits(args, examples, train_sz, dev_sz, subset=''):
    trname = args.outfile.replace('.csv', f'_{subset}_train.csv' if subset else '_train.csv')
    dvname = args.outfile.replace('.csv', f'_{subset}_dev.csv' if subset else '_dev.csv')
    tename = args.outfile.replace('.csv', f'_{subset}_test.csv' if subset else '_test.csv')
    allname = args.outfile.replace('.csv', f'_{subset}_all.csv' if subset else '_all.csv')
    trf = open(trname, 'w')
    dvf = open(dvname, 'w')
    tef = open(tename, 'w')
    allf = open(allname, 'w')

    trw = csv.writer(trf)
    trw.writerow(['whole', 'part', 'jj', 'label', 'bin_label', 'responses'])
    dvw = csv.writer(dvf)
    dvw.writerow(['whole', 'part', 'jj', 'label', 'bin_label', 'responses'])
    tew = csv.writer(tef)
    tew.writerow(['whole', 'part', 'jj', 'label', 'bin_label', 'responses'])
    allw = csv.writer(allf)
    allw.writerow(['whole', 'part', 'jj', 'label', 'bin_label', 'responses'])
    
    random.shuffle(examples)
    for i,example in enumerate(examples):
        rank = i / len(examples)
        if rank < train_sz:
            trw.writerow(example)
        elif rank < train_sz + dev_sz:
            dvw.writerow(example)
        else:
            tew.writerow(example)
        allw.writerow(example)

    trf.close()
    dvf.close()
    tef.close()
    allf.close()

def main(args):
    split = [int(s)/100 for s in args.split.split('/')]
    print(split)
    train_sz, dev_sz, test_sz = split

    str2score = {'guaranteed': 4, 'probably': 3, 'unrelated': 2, 'unlikely': 1, 'impossible': 0}
    str2score2 = {'guaranteed': 1, 'probably': 1, 'unrelated': 0, 'unlikely': 0, 'impossible': 0}
    if args.pjj_equals_impossible:
        str2score['pjj-nonsense'] = 0
        str2score2['pjj-nonsense'] = 0

    examples = []
    maj_examples = []
    no_imp_examples = []
    maj_no_imp_examples = []
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
            examples.append([*triple, label, bin_label, *responses])
            if 0 not in responses and len(set(responses)) < len(responses):
                maj_no_imp_examples.append([*triple, label, bin_label, *responses])
                maj_examples.append([*triple, label, bin_label, *responses])
                no_imp_examples.append([*triple, label, bin_label, *responses])
            elif len(set(responses)) < len(responses):
                maj_examples.append([*triple, label, bin_label, *responses])
            elif 0 not in responses:
                no_imp_examples.append([*triple, label, bin_label, *responses])

    write_splits(args, examples, train_sz, dev_sz)
    write_splits(args, maj_examples, train_sz, dev_sz, 'maj')
    write_splits(args, no_imp_examples, train_sz, dev_sz, 'no_imp')
    write_splits(args, maj_no_imp_examples, train_sz, dev_sz, 'maj_no_imp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to AMT data')
    parser.add_argument('outfile', type=str, help='path to write annotated data. base name: will append _train, _dev, _test')
    parser.add_argument('split', type=str, help='train/test/dev split, given as slash delimited percentages like 70/10/20')
    parser.add_argument('--pjj-equals-impossible', const=True, action='store_const', help='flag to let pjj-nonsense be equal to impossible')
    args = parser.parse_args()
    main(args)

