#!/bin/bash
# Loop from 1 to 10

###dtd

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/cross_dataset.yaml --test_config_path few_shot_no_val_lr1e-5_lambda_1_N --dataset_config configs/dtd.yaml --seed $i --shots 4
done


###caltech101

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/cross_dataset.yaml --test_config_path few_shot_no_val_lr1e-5_lambda_1_N --dataset_config configs/caltech101.yaml --seed $i --shots 4
done

###ucf101

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/cross_dataset.yaml --test_config_path few_shot_no_val_lr1e-5_lambda_1_N --dataset_config configs/ucf101.yaml --seed $i --shots 4
done


###oxford_flowers

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/cross_dataset.yaml --test_config_path few_shot_no_val_lr1e-5_lambda_1_N --dataset_config configs/oxford_flowers.yaml --seed $i --shots 4
done

###stanford_cars

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/cross_dataset.yaml --test_config_path few_shot_no_val_lr1e-5_lambda_1_N --dataset_config configs/stanford_cars.yaml --seed $i --shots 4
done


###eurosat

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/cross_dataset.yaml --test_config_path few_shot_no_val_lr1e-5_lambda_1_N --dataset_config configs/eurosat.yaml --seed $i --shots 4
done

###oxford_pets

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/cross_dataset.yaml --test_config_path few_shot_no_val_lr1e-5_lambda_1_N --dataset_config configs/oxford_pets.yaml --seed $i --shots 4
done

###food101

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/cross_dataset.yaml --test_config_path few_shot_no_val_lr1e-5_lambda_1_N --dataset_config configs/food101.yaml --seed $i --shots 4
done

##fgvc

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/cross_dataset.yaml --test_config_path few_shot_no_val_lr1e-5_lambda_1_N --dataset_config configs/fgvc.yaml --seed $i --shots 4
done

##sun397

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/cross_dataset.yaml --test_config_path few_shot_no_val_lr1e-5_lambda_1_N --dataset_config configs/sun397.yaml --seed $i --shots 4
done