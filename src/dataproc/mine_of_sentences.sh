#!/bin/bash
### Use tgrep to search for *sentences* following "of [det]" patterns in tgrep corpora files
### Will use N of your cores
N=20
for fn in /auto/nlg-05/jgm_234/commonsense-part-whole/data/gigaword5-treebank/*.t2c; do 
    ((i=i%N)); ((i++==0)) && wait
    (out_fn="${fn/.t2c/.of.pwsents}"
    /auto/nlg-05/jgm_234/TGrep2/tgrep2 -c $fn -a -t -d -w "NP < ((NP << NN|NNS) . (PP < (IN < of . (NP < DT <\` NN|NNS))))" > $out_fn) & 
done

