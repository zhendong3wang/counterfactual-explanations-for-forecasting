#!/bin/bash
for seed in 1 9 30 33 39

# conda activate forecastcf

do
    echo "Random seed: $seed"
    python src/cf_search.py --dataset cif2016 --horizon 12 --back-horizon 15 --split-size 0.6 0.2 0.2 --stride-size 1 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 12 --random-seed $seed --output output.csv

    python src/cf_search.py --dataset nn5 --horizon 56 --back-horizon 70 --split-size 0.6 0.2 0.2 --stride-size 20 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 2 --ablation-horizon 56 --random-seed $seed --output output.csv

    python src/cf_search.py --dataset tourism --horizon 24 --back-horizon 30 --split-size 0.6 0.2 0.2 --stride-size 20 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1.5 --ablation-horizon 24 --random-seed $seed --output output.csv

    python src/cf_search.py --dataset sp500 --horizon 60 --back-horizon 120 --split-size 0.7 0.15 0.15 --stride-size 60 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 60 --random-seed $seed --output output.csv

    python src/cf_search.py --dataset m4 --horizon 18 --back-horizon 27 --split-size 0.7 0.15 0.15 --stride-size 45 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 2 --ablation-horizon 18 --random-test --random-seed $seed --output output.csv

    python src/cf_search.py --dataset mimic --horizon 8 --back-horizon 24 --split-size 1 0 0 --stride-size 8 --center median --desired-shift 0 --desired-change 0.1 --poly-order 1 --fraction-std 1 --ablation-horizon 8 --random-seed $seed --output output.csv

done





