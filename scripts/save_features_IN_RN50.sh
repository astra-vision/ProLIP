#!/bin/bash
# Loop from 1 to 10

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main_imagenet.py --base_config configs/experiments/save_feat.yaml --dataset_config configs/imagenet.yaml --seed $i --shots 4
done