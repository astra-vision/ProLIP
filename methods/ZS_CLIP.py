import torch
from .method import FSCLIPmethod


from clip import clip

from .coop import load_clip_to_cpu
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


CUSTOM_TEMPLATES = {
    # "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordPets": "a type of pet, a photo of a {}.",
    # "OxfordFlowers": "a photo of a {}, a type of flower.",
    "OxfordFlowers": "a type of flower, a photo of a {}.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    # "Food101": "a photo of {}, a type of food.",
    "Food101": "a type of food, a photo of {}.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

IMAGENET_TEMPLATES_SELECT = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

class ZeroshotCLIP(FSCLIPmethod):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1,
                                                               keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


class ZeroshotCLIP2(FSCLIPmethod):
    """Prompt ensembling."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.lr = args['lr']
        self.epoch = args['train_epoch']
        self.cfg = args

    templates = IMAGENET_TEMPLATES_SELECT

    def forward(self,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                test_loader_v2 : torch.utils.data.DataLoader,
                test_loader_sketch : torch.utils.data.DataLoader,
                test_loader_a : torch.utils.data.DataLoader,
                test_loader_r: torch.utils.data.DataLoader,
                text_weights: torch.tensor,
                text_weights_a: torch.tensor,
                text_weights_r: torch.tensor,
                text_weights_before: torch.tensor,
                texts: torch.tensor,
                model: nn.Module,
                state_dict,
                classnames,
                task: int,
                shots: int):

        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p)
                                 for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1,
                                                               keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(
            dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model

        acc_test = self.compute_image_features_test(clip_model, test_loader,text_weights)
        acc_test_v2 = self.compute_image_features_test(clip_model, test_loader_v2,text_weights)
        acc_test_s = self.compute_image_features_test(clip_model, test_loader_sketch,text_weights)
        acc_test_a = self.compute_image_features_test(clip_model, test_loader_a,text_weights_a)
        acc_test_r = self.compute_image_features_test(clip_model, test_loader_r,text_weights_r)

        if cfg['dataset'] == 'imagenet':
            return acc_test, acc_test_v2, acc_test_s, acc_test_a, acc_test_r


    def compute_image_features_test(self,clip_model, loader,text_weights):

        acc_list = []
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                images, target = images.cuda(), target.cuda()
                x_pre_proj, image_features = clip_model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)

                logits_test = 100. * image_features @ text_weights
                acc = logits_test.argmax(dim=1).cpu() ==  target.cpu()
                acc_list += acc.detach().tolist()

            acc_test = 100 * np.mean(acc_list)
        return acc_test

