#!/bin/bash

# python src/cf_search.py --dataset cif2016 --horizon 12 --back-horizon 15 --split-size 0.6 0.2 0.2 --stride-size 1 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 12 --output output_horizon.csv

python src/cf_search.py --dataset cif2016 --horizon 12 --back-horizon 15 --split-size 0.6 0.2 0.2 --stride-size 1 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 6 --output output_horizon.csv

python src/cf_search.py --dataset cif2016 --horizon 12 --back-horizon 15 --split-size 0.6 0.2 0.2 --stride-size 1 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 3 --output output_horizon.csv

python src/cf_search.py --dataset cif2016 --horizon 12 --back-horizon 15 --split-size 0.6 0.2 0.2 --stride-size 1 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 1 --output output_horizon.csv

# python src/cf_search.py --dataset nn5 --horizon 56 --back-horizon 70 --split-size 0.6 0.2 0.2 --stride-size 20 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 2 --ablation-horizon 56 --output output_horizon.csv

python src/cf_search.py --dataset nn5 --horizon 56 --back-horizon 70 --split-size 0.6 0.2 0.2 --stride-size 20 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 2 --ablation-horizon 28 --output output_horizon.csv

python src/cf_search.py --dataset nn5 --horizon 56 --back-horizon 70 --split-size 0.6 0.2 0.2 --stride-size 20 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 2 --ablation-horizon 14 --output output_horizon.csv

python src/cf_search.py --dataset nn5 --horizon 56 --back-horizon 70 --split-size 0.6 0.2 0.2 --stride-size 20 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 2 --ablation-horizon 7 --output output_horizon.csv

python src/cf_search.py --dataset nn5 --horizon 56 --back-horizon 70 --split-size 0.6 0.2 0.2 --stride-size 20 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 2 --ablation-horizon 1 --output output_horizon.csv

# python src/cf_search.py --dataset tourism --horizon 24 --back-horizon 30 --split-size 0.6 0.2 0.2 --stride-size 20 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1.5 --ablation-horizon 24 --output output_horizon.csv

python src/cf_search.py --dataset tourism --horizon 24 --back-horizon 30 --split-size 0.6 0.2 0.2 --stride-size 20 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1.5 --ablation-horizon 12 --output output_horizon.csv

python src/cf_search.py --dataset tourism --horizon 24 --back-horizon 30 --split-size 0.6 0.2 0.2 --stride-size 20 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1.5 --ablation-horizon 6 --output output_horizon.csv

python src/cf_search.py --dataset tourism --horizon 24 --back-horizon 30 --split-size 0.6 0.2 0.2 --stride-size 20 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1.5 --ablation-horizon 1 --output output_horizon.csv

# python src/cf_search.py --dataset sp500 --horizon 60 --back-horizon 120 --split-size 0.7 0.15 0.15 --stride-size 60 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 60 --output output_horizon.csv

python src/cf_search.py --dataset sp500 --horizon 60 --back-horizon 120 --split-size 0.7 0.15 0.15 --stride-size 60 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 30 --output output_horizon.csv

python src/cf_search.py --dataset sp500 --horizon 60 --back-horizon 120 --split-size 0.7 0.15 0.15 --stride-size 60 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 15 --output output_horizon.csv

python src/cf_search.py --dataset sp500 --horizon 60 --back-horizon 120 --split-size 0.7 0.15 0.15 --stride-size 60 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 7 --output output_horizon.csv

python src/cf_search.py --dataset sp500 --horizon 60 --back-horizon 120 --split-size 0.7 0.15 0.15 --stride-size 60 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 1 --output output_horizon.csv

# python src/cf_search.py --dataset m4 --horizon 18 --back-horizon 27 --split-size 0.7 0.15 0.15 --stride-size 45 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 2 --ablation-horizon 18 --random-test --output output_horizon.csv

python src/cf_search.py --dataset m4 --horizon 18 --back-horizon 27 --split-size 0.7 0.15 0.15 --stride-size 45 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 2 --ablation-horizon 9 --random-test --output output_horizon.csv

python src/cf_search.py --dataset m4 --horizon 18 --back-horizon 27 --split-size 0.7 0.15 0.15 --stride-size 45 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 2 --ablation-horizon 5 --random-test --output output_horizon.csv

python src/cf_search.py --dataset m4 --horizon 18 --back-horizon 27 --split-size 0.7 0.15 0.15 --stride-size 45 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 2 --ablation-horizon 1 --random-test --output output_horizon.csv

# python src/cf_search.py --dataset mimic --horizon 8 --back-horizon 24 --split-size 1 0 0 --stride-size 8 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 8 --output output_horizon.csv

python src/cf_search.py --dataset mimic --horizon 8 --back-horizon 24 --split-size 1 0 0 --stride-size 8 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 4 --output output_horizon.csv

python src/cf_search.py --dataset mimic --horizon 8 --back-horizon 24 --split-size 1 0 0 --stride-size 8 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 2 --output output_horizon.csv

python src/cf_search.py --dataset mimic --horizon 8 --back-horizon 24 --split-size 1 0 0 --stride-size 8 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 1 --output output_horizon.csv

