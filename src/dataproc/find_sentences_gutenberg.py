"""
    Look thru all downloaded gutenberg books for sentences with [adj whole]
"""
import csv, os, re, sys

from nltk.tokenize import sent_tokenize

whole_jjs = set()
with open(sys.argv[1]) as f:
    r = csv.reader(f)
    #header
    next(r)
    for row in r:
        whole_jjs.add((row[0], row[2]))

BASE_DIR = '../../data/gutenberg/www.gutenberg.org/robot/'
for f in os.listdir(BASE_DIR):
    if f.endswith('.txt'):
        book = open('%s/%s' % (BASE_DIR, f)).read().replace('\n', ' ')
        #get rid of multiple whitespace
        book = re.sub('\s+', ' ', book)
        sents = sent_tokenize(book)
        for sent in sents:
            #TODO: look for word
            pass

        
