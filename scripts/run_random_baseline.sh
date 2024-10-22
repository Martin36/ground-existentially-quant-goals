#!/bin/bash

set -e

PYTHONPATH=$PYTHONPATH:src

# Blocks-clear
echo "Running random baseline on blocks-clear"
python src/test.py \
  --model_type=val_sub \
  --random_sub \
  --data_file=data/datasets/blocks-clear/6v-9m-500_mc/dev.json \
  --batch_size=1

python src/stats/paper_stats.py \
  --input_file=models/val_sub/blocks-clear/6v-9m-500_mc/predictions.json

# Blocks-on
echo "Running random baseline on blocks-on"
python src/test.py \
  --model_type=val_sub \
  --random_sub \
  --data_file=data/datasets/blocks-on/6v-9m-500_mc/dev.json \
  --batch_size=1

python src/stats/paper_stats.py \
  --input_file=models/val_sub/blocks-on/6v-9m-500_mc/predictions.json

# Gripper
echo "Running random baseline on gripper"
python src/test.py \
  --model_type=val_sub \
  --random_sub \
  --data_file=data/datasets/gripper/6v-15m-500_mc/dev.json \
  --batch_size=1

python src/stats/paper_stats.py \
  --input_file=models/val_sub/gripper/6v-15m-500_mc/predictions.json

# Delivery
echo "Running random baseline on delivery"
python src/test.py \
  --model_type=val_sub \
  --random_sub \
  --data_file=data/datasets/delivery/6v-35m-500_mc/dev.json \
  --batch_size=1

python src/stats/paper_stats.py \
  --input_file=models/val_sub/delivery/6v-35m-500_mc/predictions.json

# Visitall
echo "Running random baseline on visitall"
python src/test.py \
  --model_type=val_sub \
  --random_sub \
  --data_file=data/datasets/visitall/6v-20m-500_mc/dev.json \
  --batch_size=1

python src/stats/paper_stats.py \
  --input_file=models/val_sub/visitall/6v-20m-500_mc/predictions.json

# Merge paper stats csv files
echo "Merging paper stats csv files"
python src/stats/merge_csv.py \
  --input_files=models/val_sub/blocks-clear/6v-9m-500_mc/paper_stats.csv,models/val_sub/blocks-on/6v-9m-500_mc/paper_stats.csv,models/val_sub/gripper/6v-15m-500_mc/paper_stats.csv,models/val_sub/delivery/6v-35m-500_mc/paper_stats.csv,models/val_sub/visitall/6v-20m-500_mc/paper_stats.csv