#!/bin/bash
N=20
for fn in *.t2c; do 
    ((i=i%N)); ((i++==0)) && wait
    (out_fn="${fn/.t2c/.pws}"
    /auto/nlg-05/jgm_234/TGrep2/tgrep2 -c $fn -a -t -d "NP < ((NP << \`NN|NNS) . (PP < (IN < of . (NP < DT <\` \`NN|NNS))))" | paste -s -d' \n' > $out_fn) & 
done

