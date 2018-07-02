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
        for doc in docs:
            if not doc.strip().startswith('<DOC'):continue

            #parse
            try:
                docx = xmltodict.parse(doc.replace('&', '&amp;') + "</DOC>")
            except:
                import pdb; pdb.set_trace()
                
            #get text
            docxdoc = docx['DOC']
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
                import pdb; pdb.set_trace()

            if paras is None:
                continue

            for ix,para in enumerate(paras):
                if para is None: 
                    continue
                sents = sent_tokenize(para)
                for sent in sents:
                    sent = sent.replace('\n', ' ')
                    words = word_tokenize(sent)
                    for w1, w2 in zip(words[:-1], words[1:]):
                        w1l, w2l = w1.lower(), w2.lower()
                        #consider both possiblities - red car and car red. 
                        if (w2l, w1l) in whole_jjs:
                            whole_jjs[(w2l,w1l)].add(sent)
                            sentences_found += 1
                            if sentences_found % 100 == 0:
                                print("%s: %d sentences found" % (dirname, sentences_found))
                                wjjs_with_context = len([tup for tup,snt in whole_jjs.items() if len(snt) > 0])
                                print("%s: %d whole-jjs with sentences" % (dirname, wjjs_with_context))
                        elif  (w1l, w2l) in whole_jjs:
                            whole_jjs[(w1l,w2l)].add(sent)
                            sentences_found += 1
                            if sentences_found % 100 == 0:
                                print("%s: %d sentences found" % (dirname, sentences_found))
                                wjjs_with_context = len([tup for tup,snt in whole_jjs.items() if len(snt) > 0])
                                print("%s: %d whole-jjs with sentences" % (dirname, wjjs_with_context))
            num_docs += 1
            #just write periodically idc
            if num_docs % 10 == 0:
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

