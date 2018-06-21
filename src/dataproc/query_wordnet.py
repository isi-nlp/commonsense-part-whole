#!/usr/bin/python
# Author: Nanyun Violet Peng (npeng@isi.edu)

import gzip, sys, os
from itertools import product
from nltk.corpus import wordnet as wn

def wordnet_synset_parts():
    synset_with_many_parts = dict()
    for synset in list(wn.all_synsets('n')):
        parts = synset.part_meronyms()
        if len(parts) > 1:
            synset_key = str(synset.lemma_names()[0])
            #assert synset_key not in synset_with_many_parts, (synset_key, synset_with_many_parts[synset_key])
            synset_with_many_parts[synset_key] = ','.join([str(lemma.name()) for item in parts for lemma in item.lemmas()])
    print ('the number of synsets with multiple parts:', len(synset_with_many_parts))
    for k, v in synset_with_many_parts.items():
        print(k, v)
    return synset_with_many_parts


def get_jj_nn(filename, synset_with_many_parts, outfile): 
    with gzip.open(filename) as inf:
        for line in inf:
            elems = line.decode('utf-8').strip().split(' ')
            if elems[0] in synset_with_many_parts:
                jjs = [elems[i] for i in range(1,len(elems),2) if int(elems[i+1]) > 1000]
                parts = synset_with_many_parts[elems[0]].split(',')
                outfile.write('='*20 + elems[0] + '='*20 + '\n')
                for pair in product(jjs, parts):
                    outfile.write(str(pair)+'\n')


def get_jj_nn_dir(dir_name, outfile_name, synset_with_many_parts):
    try:
        assert os.path.isdir(dir_name)
    except TypeError:
        sys.stderr.write('expecting a directory, got %s.\n' % dir_name)
    with open(outfile_name, 'w') as outf:
        for fn in os.listdir(dir_name):
            if not fn.startswith('nn.jj.amod'):
                continue
            print ('processing file:', fn)
            get_jj_nn(os.path.join(dir_name, fn), synset_with_many_parts, outf)


def main():
    synset_with_many_parts = wordnet_synset_parts()
    get_jj_nn_dir(sys.argv[1], sys.argv[2], synset_with_many_parts)

if __name__ == '__main__':
    main()
