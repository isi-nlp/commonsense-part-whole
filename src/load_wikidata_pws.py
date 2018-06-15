id2name = {}
with open('../data/wikidata/wikidata-en-lowercase-single-names.csv') as f:
    for r in f:
        row = r.split()
        nid = row[0].split('/')[-1].split('>')[0]
        name = row[2].split('"')[1]
        id2name[nid] = name
        
part2whole = defaultdict(set)
with open('../data/wikidata/wikidata-partof.csv') as f:
    for r in f:
        row = r.split()
        pid = row[0].split('/')[-1].split('>')[0]
        wid = row[2].split('/')[-1].split('>')[0]
        if pid in id2name and wid in id2name:
            part2whole[id2name[pid]].add(id2name[wid])
        
with open('../data/wikidata/wikidata-haspart.csv') as f:
    for r in f:
        row = r.split()
        wid = row[0].split('/')[-1].split('>')[0]
        pid = row[2].split('/')[-1].split('>')[0]
        if pid in id2name and wid in id2name:
            part2whole[id2name[pid]].add(id2name[wid])
        
whole2parts = defaultdict(set)
for part, wholes in part2whole.items():
    for whole in wholes:
        whole2parts[whole].add(part)
        
with open('../data/nouns/wikidata-pws.csv', 'w') as of:
    w = csv.writer(of)
    for whole, parts in whole2parts.items():
        for part in parts:
            w.writerow([whole, part])
            
