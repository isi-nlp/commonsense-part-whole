#!/bin/bash

cat ../../data/annotated/full_train.csv ../../data/annotated/full_dev.csv ../../data/annotated/full_test.csv | cut -d, -f4 | sort | uniq -c | head -5 > ../../data/annotated/distr.txt
python plot_distr.py
