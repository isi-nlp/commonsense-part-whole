#!/bin/bash
### Use tgrep to search for *noun pairs* following possessive patterns in tgrep corpora files
### Will use N of your cores
N=20
for fn in *.t2c; do
    ((i=i%N)); ((i++==0)) && wait
    (out_fn="${fn/.t2c/.pws}"
    /auto/nlg-05/jgm_234/TGrep2/tgrep2 -c $fn -a -t -d "NP < (NP < (\`NN . (POS < 's)) | < (\`NNS . (POS < '))) <- \`NN|NNS" | grep -v "<none>" | paste -s -d' \n' | awk '{print $2, $1}' >> poss_cands.txt) &
done

