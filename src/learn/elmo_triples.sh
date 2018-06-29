#!/bin/bash
###    Get elmo embeddings for each triple as if it were a sentence, to be passed into MLP
###        - Maybe we should just get context-less embeddings for individual words, but doing this for now

if [ $# -lt 2 ]; then
    echo 1>&2 "$0: not enough arguments"
    exit 1
fi
sentfile="${1/.csv/_sent.csv}"
echo "$sentfile"
cut -d, -f1,2,3 $1 | tr ',' ' ' | tail -n +2 > "$sentfile"
allennlp elmo "$sentfile" $2 --weight-file ../../lib/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 --average

devfile="${1/_train.csv/_dev.csv}"
sentfile="${devfile/.csv/_sent.csv}"
elmofile="${2/train/dev}"
echo "$sentfile"
cut -d, -f1,2,3 $devfile | tr ',' ' ' | tail -n +2 > "$sentfile"
allennlp elmo "$sentfile" $elmofile --weight-file ../../lib/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 --average

testfile="${1/_train.csv/_test.csv}"
sentfile="${testfile/.csv/_sent.csv}"
elmofile="${2/train/test}"
echo "$sentfile"
cut -d, -f1,2,3 $testfile | tr ',' ' ' | tail -n +2 > "$sentfile"
allennlp elmo "$sentfile" $elmofile --weight-file ../../lib/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 --average
