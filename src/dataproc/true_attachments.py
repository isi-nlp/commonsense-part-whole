import csv, sys

with open('../data/ngrams/nounargs/%s-%s.amod' % (sys.argv[1], sys.argv[2])) as f:
    with open('../data/ngrams/nounargs/%s-%s2.amod' % (sys.argv[1], sys.argv[2]), 'w') as of:
        r = csv.reader(f, delimiter='\t')
        w = csv.writer(of, delimiter='\t')
        for row in r:
            sent = row[0].split(':')[1].split()
            if len(sent) > 4:
                for word in sent:
                    word, _, _, head = word.split('/')
                    if word == sys.argv[1] and int(head) > 0:
                        if sent[int(head)-1].split('/')[0] == sys.argv[2]:
                            w.writerow(row)
