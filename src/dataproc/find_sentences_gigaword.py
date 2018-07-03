"""
    Look thru all gigaword docs for sentences with [adj whole]
"""
import csv, json, os, random, re, sys
from collections import OrderedDict
from multiprocessing import Pool

from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import xmltodict

def process_dir(dirname):
    whole_jjs = {}
    with open(sys.argv[1]) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            whole_jjs[(row[0], row[2])] = set()

    sentences_found = 0
    num_docs = 0
    print(dirname)
    for idx,f in enumerate(os.listdir('%s/%s' % (BASE_DIR, dirname))):
        raw = open('%s/%s/%s' % (BASE_DIR, dirname, f)).read()
        docs = raw.split('</DOC>')
        for doc_ix,doc in enumerate(docs):
            if not doc.strip().startswith('<DOC'):continue

            #parse
            try:
                docx = xmltodict.parse(doc.replace('&', '&amp;') + "</DOC>")
            except Exception as e:
                print("EXCEPTION!")
                print(dirname)
                print(f)
                print(idx)
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
                print(dirname)
                print(f)
                print(idx)
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
                        if (w2l, w1l) in whole_jjs:
                            whole_jjs[(w2l,w1l)].add(sent)
                            sentences_found += 1
                            display_update = sentences_found % 100 == 0
                        elif  (w1l, w2l) in whole_jjs:
                            whole_jjs[(w1l,w2l)].add(sent)
                            sentences_found += 1
                            display_update = sentences_found % 100 == 0
                        else:
                            #also look for multi-word wholes
                            multi_part = ' '.join((w1l,w2l))
                            if word_ix + 2 <= len(words) - 1 and (multi_part, words[word_ix + 2].lower()) in whole_jjs:
                                whole_jjs[(multi_part, words[word_ix + 2].lower())].add(sent)
                                sentences_found += 1
                                display_update = sentences_found % 100 == 0
                            elif word_ix - 1 >= 0 and (multi_part, words[word_ix - 1].lower()) in whole_jjs:
                                whole_jjs[(multi_part, words[word_ix - 1].lower())].add(sent)
                                sentences_found += 1
                                display_update = sentences_found % 100 == 0

                        if display_update:
                            print("%s: %d sentences found" % (dirname, sentences_found))
                            wjjs_with_context = len([tup for tup,snt in whole_jjs.items() if len(snt) > 0])
                            print("%s: %d whole-jjs with sentences" % (dirname, wjjs_with_context))

            num_docs += 1
            #just write periodically idc
            if num_docs % 100 == 0:
                with open('../../data/candidates/wj-gigaword-sentences-%s.json' % dirname, 'w') as of:
                    #put whole-jj into a single string and do set->list so the json works
                    wjj = {', '.join(tup):list(snts) for tup,snts in whole_jjs.items()}
                    json.dump(wjj, of, indent=1)
 
    with open('../../data/candidates/wj-gigaword-sentences-%s.json' % dirname, 'w') as of:
        #put whole-jj into a single string and do set->list so the json works
        wjj = {', '.join(tup):list(snts) for tup,snts in whole_jjs.items()}
        json.dump(wjj, of, indent=1)
    return whole_jjs

if __name__ == "__main__":

    BASE_DIR = '../../../e/gigaword_eng_5/data'

    dirs = os.listdir(BASE_DIR)
    whole_jjs = {}
    with open(sys.argv[1]) as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            whole_jjs[(row[0], row[2])] = set()

    pool = Pool(processes=7)
    for dir_jjs in pool.imap_unordered(process_dir, dirs):
        for tup,snts in dir_jjs.items():
            whole_jjs[tup].update(snts)

    with open('../../data/candidates/wj-gigaword-sentences.json', 'w') as of:
        #put whole-jj into a single string and do set->list so the json works
        wjj = {', '.join(tup):list(snts) for tup,snts in whole_jjs.items()}
        json.dump(wjj, of, indent=1)

