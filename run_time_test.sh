#!/bin/bash

python src/cf_search.py --dataset cif2016 --horizon 12 --back-horizon 15 --split-size 0.6 0.2 0.2 --stride-size 1 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 12 --runtime-test --output output_runtime.csv
