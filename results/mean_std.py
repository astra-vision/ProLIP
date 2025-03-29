import numpy as np
import statistics
import argparse
from pathlib import Path

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acc_path', default='', help='Path to .txt files of accuracy')
    return parser.parse_args()

def mean_acc(file_path):
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file]
    mean, std_dev = np.mean(numbers), np.std(numbers)
    print(f"{file_path.name}: Mean={mean}, Std Dev={std_dev}")
    return mean, std_dev

def mean_per_dataset(main_paths, datasets, shots):
    means, stds = {i: [] for i in shots}, {i: [] for i in shots}
    
    for dataset in datasets:
        print(dataset)
        for i in shots:
            for main_path in main_paths:
                file_path = Path(main_path) / f"{dataset}{i}_shot.txt"
                m, s = mean_acc(file_path)
                means[i].append(m)
                stds[i].append(s)
    
    for i in shots:
        print(f"Mean for {i}-shot: {statistics.mean(means[i])}")

if __name__ == "__main__":
    args = get_arguments()
    main_paths = [args.acc_path]
    datasets = ["imagenet", "sun397", "dtd", "caltech101", "ucf101", "oxford_flowers", 
                "stanford_cars", "eurosat", "oxford_pets", "food101", "fgvc"]
    shots = [1, 2, 4, 8, 16]
    
    mean_per_dataset(main_paths, datasets, shots)
