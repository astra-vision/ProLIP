import os
from pathlib import Path
import random
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import build_dataset
from datasets.imagenet import ImageNet
from datasets.utils import build_data_loader
import clip
from methods import __dict__ as all_methods
from utils import *
from methods.utils import *


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_config', default='configs/base.yaml',
        help='setting of Few-shot CLIP')
    parser.add_argument(
        '--test_config_path', type=str, default=None,
        help='path to tested checkpoint')
    parser.add_argument(
        '--dataset_config', default='configs/imagenet.yaml',
        help='dataset config')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--seed", type=int, default=1,
                    help="seed of support set(default: 1)")
    parser.add_argument("--shots", type=int, default=1,
                        help="number of shots(default: 1)")
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.dataset_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg, args



class RelabeledDataset(Dataset):
    def __init__(self, subset, relabeler):
        self.subset = subset
        self.relabeler = relabeler

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        new_label = self.relabeler.get(label, label)
        return image, new_label


def main():

    # Load config file
    cfg, args = get_arguments()

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    method = all_methods[cfg['method']](args=cfg)

    # CLIP
    state_dict, clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)

    print("Preparing ImageNet dataset.")
    subsample = cfg["SUBSAMPLE_CLASSES"]
    imagenet = ImageNet(cfg['root_path'], preprocess)
    
    classnames = imagenet.classnames

    if cfg["SUBSAMPLE_CLASSES"] == "base":
        classnames = classnames[:500]
    elif cfg["SUBSAMPLE_CLASSES"] == "new":
        classnames = classnames[500:]

    domain_shift_data = ["imagenetv2", "imagenet_sketch", "imagenet_rendition", "imagenet_adversarial"]
    print(cfg["dataset"])
    
    if cfg["SUBSAMPLE_CLASSES"] == "all":
        if cfg["dataset"] in domain_shift_data:
            print("Preparing target dataset.")
            dataset = build_dataset(cfg['dataset'], subsample, cfg['root_path'])

            test_loader = build_data_loader(data_source=dataset.test, batch_size=100, is_train=False, tfm=preprocess, shuffle=False)
        else:
            dataset = build_dataset("imagenetv2",subsample, cfg['root_path'])
            test_loader_v2 = build_data_loader(data_source=dataset.test, batch_size=100, is_train=False, tfm=preprocess, shuffle=False)

            dataset = build_dataset("imagenet_sketch", subsample, cfg['root_path'])
            test_loader_sketch = build_data_loader(data_source=dataset.test, batch_size=100, is_train=False, tfm=preprocess, shuffle=False)

            dataset = build_dataset("imagenet_adversarial", subsample, cfg['root_path'])
            test_loader_a = build_data_loader(data_source=dataset.test, batch_size=100, is_train=False, tfm=preprocess, shuffle=False)
            classnames_a = dataset.classnames

            dataset = build_dataset("imagenet_rendition",subsample, cfg['root_path'])
            test_loader_r = build_data_loader(data_source=dataset.test, batch_size=100, is_train=False, tfm=preprocess, shuffle=False)
            classnames_r = dataset.classnames

            test_loader = torch.utils.data.DataLoader(
                imagenet.test, batch_size=100, num_workers=8, shuffle=False)


    else:

        half_size = len(imagenet.test) // 2
        if cfg["SUBSAMPLE_CLASSES"] == "base":
            test_set = torch.utils.data.Subset(imagenet.test, range(half_size))
        elif cfg["SUBSAMPLE_CLASSES"] == "new":
            test_set = torch.utils.data.Subset(imagenet.test, range(half_size, len(imagenet.test)))
        
        # Apply relabeling
        if cfg["SUBSAMPLE_CLASSES"] == "new":
            labels = sorted(set(imagenet.test.targets[half_size:]))
            relabeler = {y: y_new for y_new, y in enumerate(labels)}
            test_set = RelabeledDataset(test_set, relabeler)

        test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=100, num_workers=8, shuffle=False)
        
        test_loader_v2 = []
        test_loader_sketch = []
        test_loader_a = []
        test_loader_r = []


    # Textual features
    print("Getting textual features as CLIP's classifier.")
    texts, clip_weights_before, clip_weights = clip_classifier(
        classnames, imagenet.template, clip_model)
    if cfg["SUBSAMPLE_CLASSES"] == "all":
        _, clip_weights_before_a, clip_weights_a = clip_classifier(
            classnames_a, imagenet.template, clip_model)
        _, clip_weights_before_r, clip_weights_r = clip_classifier(
            classnames_r, imagenet.template, clip_model)
    else:
        clip_weights_a = []
        clip_weights_r = []

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Start Training Task:{}".format(str(args.seed)))
    # few_shot_train_data, few_shot_val_data  = imagenet.train, imagenet.val
    few_shot_train_data, few_shot_val_data  = imagenet.generate_fewshot_dataset(args.shots)
    
    if cfg['shuffle']:
        train_loader = torch.utils.data.DataLoader(
            few_shot_train_data, batch_size=256, num_workers=8, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            few_shot_train_data, batch_size=cfg["batch_size"], num_workers=8, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
            few_shot_val_data, batch_size=cfg["batch_size"], num_workers=8, shuffle=False)


    if cfg['save_features']:

        if cfg['backbone'] == "ViT-B/16":
            backbone_name = "ViTB16"
        elif cfg['backbone'] == "ViT-B/32":
            backbone_name = "ViTB32"
        else:
            backbone_name = cfg['backbone']

        for i in range (cfg['aug_views']):
            print(i)
            train_x_before_proj, train_labels = compute_image_features(clip_model, train_loader)
            save_path_features = Path(cfg['root_path']) / f"features_{backbone_name}_{cfg['dataset']}" / f"{args.shots}_shot" / f"seed{args.seed}" / f"f{i}.pth" #(N*K,Do)
            save_path_features.parent.mkdir(parents=True, exist_ok=True)
            torch.save(train_x_before_proj, save_path_features)
            if i==0:
                save_path_labels = Path(cfg['root_path']) / f"features_{backbone_name}_{cfg['dataset']}" / f"{args.shots}_shot" / f"seed{args.seed}" / "label.pth"
                save_path_labels.parent.mkdir(parents=True, exist_ok=True)
                torch.save(train_labels, save_path_labels)
        assert(0)
    
    path = Path(args.base_config)
    config_file = path.stem

    test_config_path = args.test_config_path

    loss, acc, acc_v2,acc_s, acc_a, acc_r = method(train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    test_loader_v2 = test_loader_v2,
                    test_loader_sketch = test_loader_sketch,
                    test_loader_a = test_loader_a,
                    test_loader_r = test_loader_r,
                    text_weights=clip_weights,
                    text_weights_a = clip_weights_a,
                    text_weights_r = clip_weights_r,
                    text_weights_before = clip_weights_before,
                    model=clip_model,
                    state_dict = state_dict,
                    classnames=classnames,
                    task=args.seed,
                    shots=args.shots,
                    config_file= config_file,
                    test_config_path= test_config_path)

    
    print('Final Accuracy on task {}: {}'.format(str(args.seed), acc))
    append_to_file(Path("results") / config_file / f"{cfg['dataset']}{args.shots}_shot.txt", str(acc))

    if cfg["SUBSAMPLE_CLASSES"] == "all":
        file_paths = [
                    Path("results") / config_file / f"{cfg['dataset']}_v2_{args.shots}_shot.txt",
                    Path("results") / config_file / f"{cfg['dataset']}_s_{args.shots}_shot.txt",
                    Path("results") / config_file / f"{cfg['dataset']}_a_{args.shots}_shot.txt",
                    Path("results") / config_file / f"{cfg['dataset']}_r_{args.shots}_shot.txt",
                        ]
        values = [acc_v2, acc_s, acc_a, acc_r]

        for file_path, value in zip(file_paths, values):
            append_to_file(file_path, str(value))


def write_to_csv(cfg, path, test_stats):
    
    try:
        res = pd.read_csv(path)
    except:
        res = pd.DataFrame()
    records = res.to_dict('records')
    if cfg['method'] == "TIPAdapter" and cfg["finetune"]:
        test_stats['method'] = "TIPAdapter-F"
    else:
        test_stats['method'] = cfg['method']
    test_stats['acc'] = round(test_stats['acc'],4)
    test_stats['std'] = round(test_stats['std'],4)
    test_stats['num_shots'] = cfg['shots']
    test_stats['tasks'] = cfg['tasks']

    records.append(test_stats)
    # Save back to dataframe
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False)

if __name__ == '__main__':
    main()

