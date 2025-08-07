#!/bin/bash

for shot in 2 8 16 32; do
  for seed in {1..10}; do
    python CMA_fewshot.py \
      --seed $seed \
      --dataset_name 'weibo' \
      --train_csv "./datasets/weibo/weibo_train.csv" \
      --test_csv "./datasets/weibo/weibo_test.csv" \
      --img_path "./datasets/weibo/all_images/" \
      --shot $shot \
      --save_path "./saved_adapter/weibo_shot${shot}_seed${seed}"
  done
done
