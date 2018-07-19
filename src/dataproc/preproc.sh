#!/bin/bash
for fn in *.txt; do ( 
    #tmp_fn="${fn/.bz2/.txt}"
    tc_fn="${fn/.txt/.t2c}"
    #bzip2 -dc $fn | tail -n +4 | head -n -2 | sed 's:<\(/\|\)parse>\( \|\)::g' | sed 's/ROOT/TOP/g' > $tmp_fn
    time /auto/nlg-05/jgm_234/TGrep2/tgrep2 -p $fn $tc_fn
    rm $fn ) &
done
