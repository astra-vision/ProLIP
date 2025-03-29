#!/bin/bash
# Loop from 1 to 10

###dtd

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/dtd.yaml --seed $i --shots 1
done


for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/dtd.yaml --seed $i --shots 2
done


for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/dtd.yaml --seed $i --shots 4
done


for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/dtd.yaml --seed $i --shots 8
done


for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/dtd.yaml --seed $i --shots 16
done


###caltech101

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/caltech101.yaml --seed $i --shots 1
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/caltech101.yaml --seed $i --shots 2
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/caltech101.yaml --seed $i --shots 4
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/caltech101.yaml --seed $i --shots 8
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/caltech101.yaml --seed $i --shots 16
done

###ucf101

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/ucf101.yaml --seed $i --shots 1
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/ucf101.yaml --seed $i --shots 2
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/ucf101.yaml --seed $i --shots 4
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/ucf101.yaml --seed $i --shots 8
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/ucf101.yaml --seed $i --shots 16
done



###oxford_flowers

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/oxford_flowers.yaml --seed $i --shots 1
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/oxford_flowers.yaml --seed $i --shots 2
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/oxford_flowers.yaml --seed $i --shots 4
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/oxford_flowers.yaml --seed $i --shots 8
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/oxford_flowers.yaml --seed $i --shots 16
done


###stanford_cars

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/stanford_cars.yaml --seed $i --shots 1
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/stanford_cars.yaml --seed $i --shots 2
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/stanford_cars.yaml --seed $i --shots 4
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/stanford_cars.yaml --seed $i --shots 8
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/stanford_cars.yaml --seed $i --shots 16
done


###eurosat

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/eurosat.yaml --seed $i --shots 1
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/eurosat.yaml --seed $i --shots 2
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/eurosat.yaml --seed $i --shots 4
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/eurosat.yaml --seed $i --shots 8
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/eurosat.yaml --seed $i --shots 16
done

###oxford_pets

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/oxford_pets.yaml --seed $i --shots 1
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/oxford_pets.yaml --seed $i --shots 2
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/oxford_pets.yaml --seed $i --shots 4
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/oxford_pets.yaml --seed $i --shots 8
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/oxford_pets.yaml --seed $i --shots 16
done


###food101

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/food101.yaml --seed $i --shots 1
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/food101.yaml --seed $i --shots 2
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/food101.yaml --seed $i --shots 4
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/food101.yaml --seed $i --shots 8
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/food101.yaml --seed $i --shots 16
done


##fgvc

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/fgvc.yaml --seed $i --shots 1
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/fgvc.yaml --seed $i --shots 2
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/fgvc.yaml --seed $i --shots 4
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/fgvc.yaml --seed $i --shots 8
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/fgvc.yaml --seed $i --shots 16
done


##sun397

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/sun397.yaml --seed $i --shots 1
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/sun397.yaml --seed $i --shots 2
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/sun397.yaml --seed $i --shots 4
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/sun397.yaml --seed $i --shots 8
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/sun397.yaml --seed $i --shots 16
done


#imagenet
for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main_imagenet.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/imagenet.yaml --seed $i --shots 1
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main_imagenet.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/imagenet.yaml --seed $i --shots 2
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main_imagenet.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/imagenet.yaml --seed $i --shots 4
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main_imagenet.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/imagenet.yaml --seed $i --shots 8
done

for i in {1..10}
do
  # Execute the Python command with the current value of i
  python main_imagenet.py --base_config configs/experiments/few_shot_no_val_lr1e-3_lambda_1e-2.yaml --dataset_config configs/imagenet.yaml --seed $i --shots 16
done