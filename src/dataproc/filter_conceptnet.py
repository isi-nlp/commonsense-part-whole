"""
    Filters conceptnet to english language part-of relations
"""
import csv
import json

en_srcs = set(['/d/conceptnet/4/en', '/d/dbpedia/en', '/d/wiktionary/en', '/d/wordnet/3.1'])

with open('../../../data/conceptnet/conceptnet-assertions-5.6.0.csv') as f:
    with open('../../../data/conceptnet/conceptnet-partof-en.csv', 'w') as of:
        w = csv.writer(of, delimiter='\t')
        r = csv.reader(f, delimiter='\t')
        for row in r:
            if row[1] == '/r/PartOf':
                src = json.loads(row[-1])['dataset']
                if src in en_srcs:
                    w.writerow(row)

