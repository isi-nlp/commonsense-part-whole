"""
    Get counts of all (case-insensitive) nouns from Google's unigram corpora
    Ignores tokens with non-alphabetic characters
"""

import argparse
from collections import Counter
import csv
from multiprocessing import Pool
import operator
import re
import os

from tqdm import tqdm

def noun_count_from_unigram_file(fname):
    #should also include hyphen?
    alpha_only = re.compile('^[a-zA-Z]+$')
    noun_counts = Counter()
    print("reading file %s" % fname)
    with open(fname) as f:
        r = csv.reader(f, delimiter='\t')
        for i,row in enumerate(r):
            if row[0].endswith('_NOUN') and int(row[1]) >= 1950:
                word = row[0].split('_NOUN')[0].lower()
                if alpha_only.match(word):
                    noun_counts[word] += int(row[2])
    return noun_counts

if __name__ == "__main__":
    #iterate over all lettered filenames
    fnames = ['../../data/ngrams/unigrams_fic/googlebooks-eng-fiction-all-1gram-20120701-%c' % chr(l) for l in range(ord('a'), ord('z')+1)]

    pool = Pool(processes=os.cpu_count())
    noun_counts = Counter()
    for alpha_counts in pool.imap_unordered(noun_count_from_unigram_file, fnames):
        noun_counts.update(alpha_counts)

    with open('../../data/ngrams/unigrams_fic/noun_counts_1950.csv', 'w') as of:
        for noun, count in sorted(noun_counts.items(), key=operator.itemgetter(0)):
            of.write(','.join([noun, str(count)]) + '\n')
