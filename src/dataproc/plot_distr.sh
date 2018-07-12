#!/bin/bash

cat ../../data/annotated/full_two_agree_train.csv ../../data/annotated/full_two_agree_dev.csv ../../data/annotated/full_two_agree_test.csv | cut -d, -f4 | sort | uniq -c | head -5 > ../../data/annotated/distr_two_agree.txt
python plot_distr.py
