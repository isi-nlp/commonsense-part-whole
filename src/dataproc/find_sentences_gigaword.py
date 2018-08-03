"""
    Look thru all gigaword docs for sentences with [adj whole/part] or [whole/part adj]
"""
import argparse, csv, json, os, random, re, sys
from collections import OrderedDict
from multiprocessing import Pool

from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import xmltodict

def process_file(fname):
    noun_jjs = {}
    with open(args.file) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            if args.noun == 'part':
                noun_jjs[(row[1], row[2])] = set()
            else:
                noun_jjs[(row[0], row[2])] = set()

    sentences_found = 0
    num_docs = 0
    print(fname)

    raw = open(f'../../data/gigaword_eng_5/data/{fname}').read()
    docs = raw.split('</DOC>')
    for doc_ix,doc in enumerate(docs):
        if not doc.strip().startswith('<DOC'):continue

        #parse
        try:
            docx = xmltodict.parse(doc.replace('&', '&amp;') + "</DOC>")
        except Exception as e:
            print("EXCEPTION!")
            print(fname)
            print(f)
            print(doc_ix)
            print(e)
            print()
            continue
            #import pdb; pdb.set_trace()
            
        #get text
        try:
            docxdoc = docx['DOC']
        except Exception as e:
            print("EXCEPTION!")
            print(fname)
            print(f)
            print(doc_ix)
            print(e)
            print()
            continue
        if 'TEXT' not in docxdoc or docxdoc['TEXT'] is None:
            continue
        text = docx['DOC']['TEXT']
        if type(text) is str:
            paras = text.split('\n\n')
        elif type(text) is OrderedDict:
            paras = text['P']
        elif text is None:
            continue
        else:
            print(type(text))
            print("idk how to deal with this. skipping")
            continue
            #import pdb; pdb.set_trace()

        if paras is None:
            continue

        for ix,para in enumerate(paras):
            if para is None: 
                continue
            sents = sent_tokenize(para)
            for sent in sents:
                sent = sent.replace('\n', ' ')
                words = word_tokenize(sent)
                for word_ix,(w1, w2) in enumerate(zip(words[:-1], words[1:])):
                    w1l, w2l = w1.lower(), w2.lower()
                    #consider both possiblities - red car and car red. 
                    display_update = False
                    if (w2l, w1l) in noun_jjs:
                        noun_jjs[(w2l,w1l)].add(sent)
                        sentences_found += 1
                        display_update = sentences_found % 100 == 0
                    elif  (w1l, w2l) in noun_jjs:
                        noun_jjs[(w1l,w2l)].add(sent)
                        sentences_found += 1
                        display_update = sentences_found % 100 == 0
                    else:
                        #also look for multi-word nouns
                        multi_part = ' '.join((w1l,w2l))
                        if word_ix + 2 <= len(words) - 1 and (multi_part, words[word_ix + 2].lower()) in noun_jjs:
                            noun_jjs[(multi_part, words[word_ix + 2].lower())].add(sent)
                            sentences_found += 1
                            display_update = sentences_found % 100 == 0
                        elif word_ix - 1 >= 0 and (multi_part, words[word_ix - 1].lower()) in noun_jjs:
                            noun_jjs[(multi_part, words[word_ix - 1].lower())].add(sent)
                            sentences_found += 1
                            display_update = sentences_found % 100 == 0

                    if display_update:
                        print("%s: %d sentences found" % (fname, sentences_found))
                        njjs_with_context = len([tup for tup,snt in noun_jjs.items() if len(snt) > 0])
                        print("%s: %d noun-jjs with sentences" % (fname, njjs_with_context))

        num_docs += 1
        #just write periodically idc
        if num_docs % 100 == 0:
            with open(filename, 'w') as of:
                #put noun-jj into a single string and do set->list so the json works
                njj = {', '.join(tup):list(snts) for tup,snts in noun_jjs.items()}
                json.dump(njj, of, indent=1)
 
    with open(filename, 'w') as of:
        #put noun-jj into a single string and do set->list so the json works
        njj = {', '.join(tup):list(snts) for tup,snts in noun_jjs.items()}
        json.dump(njj, of, indent=1)
    return noun_jjs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="triples file")
    parser.add_argument("cpu_count", type=int, help="num cpus to use")
    parser.add_argument("noun", choices=['part', 'whole'], help="look for whole-jj or part-jj?")
    args = parser.parse_args()

    BASE_DIR = '../../data/gigaword_eng_5/data'
    name = 'wj' if args.noun == 'whole' else 'pj'
    filename = f'../../data/candidates/{name}-gigaword-sentences.json'

    dirs = os.listdir(BASE_DIR)
    fnames = []
    for dirname in dirs:
        fnames.extend([os.path.join(dirname, f) for f in os.listdir('%s/%s' % (BASE_DIR, dirname))])
    noun_jjs = {}
    with open(args.file) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            if args.noun == 'part':
                noun_jjs[(row[1], row[2])] = set()
            else:
                noun_jjs[(row[0], row[2])] = set()

    pool = Pool(processes=int(args.cpu_count))
    for file_jjs in pool.imap_unordered(process_file, fnames):
        for tup,snts in file_jjs.items():
            noun_jjs[tup].update(snts)

    with open(filename, 'w') as of:
        #put noun-jj into a single string and do set->list so the json works
        njj = {', '.join(tup):list(snts) for tup,snts in noun_jjs.items()}
        json.dump(njj, of, indent=1)

