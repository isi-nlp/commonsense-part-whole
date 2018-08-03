"""
    Check out variety of wholes that share a part-adjective
"""
import pandas as pd

df = pd.read_csv('../../data/annotated/full_all.csv')

pj = df.groupby(['part', 'jj']).count()

pjf = pj[pj['whole'] > 1]
pjmults = pjf.index.values.tolist()

df['keep'] = df.apply(lambda row: tuple(row[['part', 'jj']]) in pjmults, axis=1)

dfk = df[df['keep']]
dfk = dfk.sort_values(['part', 'jj'])

numwholes = dfk.groupby(['part', 'jj']).count()['bin_label']
allsame = dfk.groupby(['part', 'jj']).apply(lambda group: all(group['bin_label'] == group['bin_label'].iloc[0]))

wholesame = pd.concat([numwholes, allsame], axis=1)
wholesame.groupby('bin_label').hist()

