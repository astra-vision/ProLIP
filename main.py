import os
from pathlib import Path
import random
import argparse
import pandas as pd
import torch
import torchvision.transforms as transforms
from datasets import build_dataset
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
        '--dataset_config', default='configs/caltech101.yaml',
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

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)

    print("Preparing dataset.")
    subsample = cfg.SUBSAMPLE_CLASSES
    dataset = build_dataset(cfg['dataset'], subsample, cfg['root_path'])
    classnames = dataset.classnames
    
    test_loader = build_data_loader(data_source=dataset.test, batch_size=100, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # Textual features
    print("Getting textual features as CLIP's classifier.")
    texts, clip_weights_before, clip_weights = clip_classifier(
        dataset.classnames, dataset.template, clip_model)
   
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Start Training Task:{}".format(str(args.seed)))
    
    few_shot_train_data = dataset.generate_fewshot_dataset_(args.shots, split="train")
    few_shot_val_data = dataset.generate_fewshot_dataset_(args.shots, split="val")

    if cfg['shuffle']:
        train_loader = build_data_loader(
            data_source=few_shot_train_data, batch_size= cfg["batch_size"], tfm=train_tranform, is_train=True, shuffle=True)
    else:
        train_loader = build_data_loader(
            data_source=few_shot_train_data, batch_size=cfg["batch_size"], tfm=train_tranform, is_train=True, shuffle=False)
    val_loader = build_data_loader(
        data_source=few_shot_val_data, batch_size=cfg["batch_size"], tfm=preprocess, is_train=False, shuffle=False)
    torch.set_printoptions(threshold=torch.inf)

    if cfg['save_features']:

        if cfg['backbone'] == "ViT-B/16":
            backbone_name = "ViTB16"
        elif cfg['backbone'] == "ViT-B/32":
            backbone_name = "ViTB32"
        else:
            backbone_name = cfg['backbone']

        for view in range (cfg['aug_views']):
            print(view)
            train_x_before_proj, train_labels = compute_image_features(clip_model, train_loader)
            save_path_features = Path(cfg['root_path']) /f"features_{backbone_name}_{cfg['dataset']}" / f"{args.shots}_shot" / f"seed{args.seed}" / f"f{view}.pth"
            save_path_features.parent.mkdir(parents=True, exist_ok=True)
            torch.save(train_x_before_proj, save_path_features) #(N*K,Do)
            if view==0:
                save_path_labels = Path(cfg['root_path']) / f"features_{backbone_name}_{cfg['dataset']}" / f"{args.shots}_shot" / f"seed{args.seed}" / "label.pth"
                save_path_labels.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
                torch.save(train_labels, save_path_labels)
        assert(0)
    
    path = Path(args.base_config)
    config_file = path.stem

    test_config_path = args.test_config_path

    loss, acc = method(train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    test_loader_v2 = None,
                    test_loader_sketch = None,
                    test_loader_a = None,
                    test_loader_r = None,
                    text_weights=clip_weights,
                    text_weights_a = None,
                    text_weights_r = None,
                    text_weights_before = clip_weights_before,
                    model=clip_model,
                    state_dict = state_dict,
                    classnames=classnames,
                    task=args.seed,
                    shots=args.shots,
                    config_file = config_file,
                    test_config_path=test_config_path)

    print('Final Accuracy on task {}: {}'.format(str(args.seed), acc))
    append_to_file(Path("results") / config_file / f"{cfg['dataset']}{args.shots}_shot.txt", str(acc))


def write_to_csv(args,cfg, path, test_stats):
    
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
    test_stats['num_shots'] = args.shots
    test_stats['tasks'] = cfg['tasks']

    records.append(test_stats)
    # Save back to dataframe
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False)

if __name__ == '__main__':
    main()

