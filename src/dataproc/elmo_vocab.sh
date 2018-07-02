#!/bin/bash
###    Get elmo embeddings for each word individually
###    Input 1: train file
###    Input 2: output file name

if [ $# -lt 2 ]; then
    echo 1>&2 "$0: not enough arguments"
    exit 1
fi
vocabfile="${1/train.csv/vocab.csv}"
echo "$vocabfile"
tail -n +2 $1 | cut -d, -f1,2,3 | tr ', ' '\n' > "$vocabfile"

devfile="${1/_train.csv/_dev.csv}"
tail -n +2 $devfile | cut -d, -f1,2,3 | tr ', ' '\n' >> "$vocabfile"
testfile="${1/_train.csv/_test.csv}"
tail -n +2 $testfile | cut -d, -f1,2,3 | tr ', ' '\n' >> "$vocabfile"

#sort in place
sort -u -o $vocabfile $vocabfile
allennlp elmo "$vocabfile" $2 --weight-file ../../lib/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 --average

