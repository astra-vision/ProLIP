import torch
import torch.nn as nn
from pathlib import Path
import argparse
from .method import FSCLIPmethod
from .utils import compute_image_features_test


class VisProjRN(nn.Module):
    def __init__(self, weight, bias):
        super(VisProjRN, self).__init__()
        input_dim, output_dim = weight.size()

        self.linear = nn.Linear(input_dim, output_dim)
        
        self.linear.weight = nn.Parameter(weight)
        self.linear.bias = nn.Parameter(bias)
        
        self.linear.weight.requires_grad =  False
        self.linear.bias.requires_grad = False
    
    def forward(self, x_before_proj):
        return self.linear(x_before_proj)

class VisProjViT(nn.Module):
    def __init__(self, vit_proj):
        super(VisProjViT, self).__init__()
       
        self.vit_proj = nn.Parameter(vit_proj)
        self.vit_proj.requires_grad = False

    def forward(self,x_before_proj):

        x = x_before_proj @ self.vit_proj
        return x


class ProLIP_eval(FSCLIPmethod):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.cfg = args

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
                model: nn.Module,
                state_dict,
                classnames,
                task: int,
                shots: int,
                config_file: str,
                test_config_path: str):

        cfg = self.cfg

        model.eval()
        
        print('Turning off gradients for all layers')
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        model.cuda()

        ## vision projection
        if cfg['backbone'] == "ViT-B/16" or cfg['backbone'] == "ViT-B/32":
            vit_proj = state_dict["visual.proj"]
            proj = VisProjViT(vit_proj)
        elif cfg['backbone'] == "RN50" or cfg['backbone'] == "RN101":    
            weight_init =  model.visual.attnpool.c_proj.weight
            bias_init = model.visual.attnpool.c_proj.bias
            proj = VisProjRN(weight_init,bias_init)  

        if cfg['cross_dataset'] == True:
            load_model_path = Path("trained_models") / test_config_path / "imagenet" / f"{shots}_shot" / f"imagenet_seed{task}.pth"
        else:
            load_model_path = Path("trained_models") / test_config_path / cfg['dataset'] / f"{shots}_shot" / f"{cfg['dataset']}_seed{task}.pth"
        proj.load_state_dict(torch.load(load_model_path))
        # Evaluation on the test set
        print('\nStart evaluation on test set')
        #########################        
        acc_test = compute_image_features_test(model, test_loader, proj, text_weights)
        # results_save_path = Path("results") / f"{cfg['dataset']}_test.txt"
        # results_save_path.parent.mkdir(parents=True,exist_ok=True)

        # with results_save_path.open("a", encoding="utf-8") as file:
        #     file.write(f"{acc_test}\n")
        
        if cfg['dataset'] == "imagenet":
            return None, acc_test, None, None, None, None
        else:
            return None, acc_test