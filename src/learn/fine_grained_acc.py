import argparse, csv, operator, os

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, mean_squared_error as mse

parser = argparse.ArgumentParser()
parser.add_argument("pred_file", type=str, help="path to preds")
args = parser.parse_args()

exp_dir = os.path.dirname(args.pred_file)
df = pd.read_csv(args.pred_file)

jj2acc = {jj:accuracy_score(df[df['jj'] == jj]['label'], df[df['jj'] == jj]['pred']) for jj in df['jj'].unique()}
jj2count = {jj: len(df[df['jj'] == jj]) for jj in df['jj'].unique()}

ranked = sorted(jj2acc.items(), key=operator.itemgetter(1))
glove_jjs = [(jj, acc, jj2count[jj]) for (jj, acc) in ranked if jj2count[jj] >= 5]
        
with open('%s/jjs-perf.csv' % exp_dir, 'w') as of:
    w = csv.writer(of)
    w.writerow(['jj', 'acc', 'count'])
    for jj, acc, count in glove_jjs:
        w.writerow([jj, acc, count])
        
part2acc = {part:accuracy_score(df[df['part'] == part]['label'], df[df['part'] == part]['pred']) for part in df['part'].unique()}
part2count = {part: len(df[df['part'] == part]) for part in df['part'].unique()}

part_ranked = sorted(part2acc.items(), key=operator.itemgetter(1))
glove_parts = [(part, acc, part2count[part]) for (part, acc) in part_ranked if part2count[part] >= 5]

with open('%s/parts-perf.csv' % exp_dir, 'w') as of:
    w = csv.writer(of)
    w.writerow(['part', 'acc', 'count'])
    for part, acc, count in glove_parts:
        w.writerow([part, acc, count])
        
whole2acc = {whole:accuracy_score(df[df['whole'] == whole]['label'], df[df['whole'] == whole]['pred']) for whole in df['whole'].unique()}
whole2count = { whole: len(df[df[ 'whole'] == whole]) for whole in df[ 'whole'].unique()}

whole_ranked = sorted(whole2acc.items(), key=operator.itemgetter(1))
glove_wholes = [( whole, acc, whole2count[whole]) for ( whole, acc) in whole_ranked if whole2count[whole] >= 5]

with open('%s/wholes-perf.csv' % exp_dir, 'w') as of:
    w = csv.writer(of)
    w.writerow(['whole', 'acc', 'count'])
    for whole, acc, count in glove_wholes:
        w.writerow([whole, acc, count])
        
