from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import numpy as np
from .method import FSCLIPmethod
from .utils import cls_acc
from .utils import compute_image_features, compute_image_features_test


class CLIPADAPTERRN(nn.Module):
    def __init__(self, weight, bias,reduction, alpha):
        super(CLIPADAPTERRN, self).__init__()
        output_dim, input_dim = weight.size()
        self.reduction = reduction
        self.alpha = alpha ### weight of zero-shot features, alpha=1 gives the zero-shot model
        ### Frozen projection layer
        self.linear = nn.Linear(output_dim, input_dim)
        self.linear.weight = nn.Parameter(weight)
        self.linear.bias = nn.Parameter(bias)
        self.linear.weight.requires_grad =  False
        self.linear.bias.requires_grad = False
        
        ### clip-adapter
        self.fc = nn.Sequential(
            nn.Linear(output_dim, output_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim // reduction, output_dim, bias=False),
            nn.ReLU(inplace=True)
        ).to('cuda').type(torch.float16)

    def forward(self, x_before_proj):
        image_features = self.linear(x_before_proj)
        x = self.fc(image_features)
        x = self.alpha * image_features + (1 - self.alpha) * x
        return x

class CLIPADAPTERViT(nn.Module):
    def __init__(self, vit_proj,reduction, alpha):
        super(CLIPADAPTERViT, self).__init__()
        _ , output_dim = vit_proj.size()
        self.reduction = reduction
        self.alpha = alpha ### weight of zero-shot features, alpha=1 gives the zero-shot model
        self.vit_proj = nn.Parameter(vit_proj)
        self.vit_proj.requires_grad = False
        
        ### clip-adapter
        self.fc = nn.Sequential(
            nn.Linear(output_dim, output_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim // reduction, output_dim, bias=False),
            nn.ReLU(inplace=True)
        ).to('cuda').type(torch.float16)

    def forward(self, x_before_proj):
        image_features = x_before_proj @ self.vit_proj
        x = self.fc(image_features)
        x = self.alpha * image_features + (1 - self.alpha) * x
        return x



class CLIPADAPTER(FSCLIPmethod):

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

        print(self.cfg['train_epoch'])
        model.eval()
        
        print('Turning off gradients for all layers')
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        model.cuda()
        mse = nn.MSELoss(reduction='sum')

        ## vision projection
        if cfg['backbone'] == "RN50" or cfg['backbone'] == "RN101":
            weight_init =  model.visual.attnpool.c_proj.weight
            bias_init = model.visual.attnpool.c_proj.bias

        elif cfg['backbone'] == "ViT-B/16" or cfg['backbone'] == "ViT-B/32":
            vit_proj = state_dict["visual.proj"]

        if cfg['backbone'] == "ViT-B/16":
            backbone_name = "ViTB16"
        elif cfg['backbone'] == "ViT-B/32":
            backbone_name = "ViTB32"
        else:
            backbone_name = cfg['backbone']

        m = len(classnames)
        train_x_before_list = []

        load_path_labels = Path(cfg['root_path']) / f"features_{backbone_name}_{cfg['dataset']}"/ f"{shots}_shot" / f"seed{task}" / "label.pth"
        train_labels = torch.load(load_path_labels)
        indices = torch.where(train_labels < m)[0]
        train_labels = train_labels[indices]

        for num in range(cfg['aug_views']):
            load_path_features = Path(cfg['root_path']) / f"features_{backbone_name}_{cfg['dataset']}" / f"{shots}_shot" / f"seed{task}" / f"f{num}.pth"
            train_x_before_view = torch.load(load_path_features)
            train_x_before_view = train_x_before_view[indices]
            train_x_before_list.append(train_x_before_view)
                
        lr_v = cfg["lr_adapter"]
        print(lr_v)


        if cfg['backbone'] == "RN50" or cfg['backbone'] == "RN101":
            adaptermlp = CLIPADAPTERRN(weight_init,bias_init,reduction=4, alpha=cfg['alpha'])
            
        elif cfg['backbone'] == "ViT-B/16" or cfg['backbone'] == "ViT-B/32":
            adaptermlp = CLIPADAPTERViT(vit_proj,reduction=4, alpha=cfg['alpha'])

        start_time = time.time()

        optimizer = torch.optim.Adam(adaptermlp.parameters(),  lr=lr_v, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'])
        
        # Train
        print('\nStart Training procedure')
            
        cnt = 0
        for train_idx in range(self.cfg['train_epoch']):
                 
            # Train
            correct_samples, all_samples = 0, 0
            loss_list_ce = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.cfg['train_epoch']))

            if (cnt+1)%self.cfg['aug_views'] == 0:
                cnt=0
            else:
                cnt+=1
            train_x_before = train_x_before_list[cnt]

            train_x_before , target = train_x_before.cuda(), train_labels.cuda()
    
            image_features = adaptermlp(train_x_before)
            image_features = F.normalize(image_features,dim=-1)

            logits = 100. * image_features @ text_weights
           
            loss = F.cross_entropy(logits, target)

            acc = cls_acc(logits, target)
            correct_samples += acc / 100 * len(logits)
            all_samples += len(logits)
            loss_list_ce.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
           
            print('Acc: {:.4f} ({:}/{:}), Loss_ce: {:.4f}'.format(correct_samples / all_samples, correct_samples, all_samples, sum(loss_list_ce)/len(loss_list_ce)))
        torch.cuda.empty_cache()
        train_x_before = train_x_before.cpu()
        train_labels = train_labels.cpu()

        del train_x_before
        del train_labels

        if cfg['save_checkpoints'] == True:
            save_path = Path("trained_models") / config_file /cfg['dataset'] / f"{shots}_shot" / f"{cfg['dataset']}_seed{task}.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(adaptermlp.state_dict(), save_path)
        
            
        # Evaluation on the test set
        print("Total time = {:.4f}".format(time.time()-start_time))        
        print('\nStart evaluation on test set')
        ##################
                
        if cfg['dataset'] == 'imagenet':
            acc_test = self.compute_image_features_test(model, test_loader,adaptermlp,text_weights)
            if cfg["SUBSAMPLE_CLASSES"] == "all":
                acc_test_v2 = compute_image_features_test(model, test_loader_v2,adaptermlp,text_weights)
                acc_test_s = compute_image_features_test(model, test_loader_sketch,adaptermlp,text_weights)
                acc_test_a = compute_image_features_test(model, test_loader_a,adaptermlp,text_weights_a)
                acc_test_r = compute_image_features_test(model, test_loader_r,adaptermlp,text_weights_r)
            else:
                acc_test_v2,acc_test_s, acc_test_a, acc_test_r = 0, 0, 0, 0

        else:
            test_features_before, test_labels = compute_image_features(model, test_loader)
            test_features = adaptermlp(test_features_before)
            test_features = F.normalize(test_features,dim=-1)
            logits_test = 100. * test_features @ text_weights

            acc_test = np.mean(logits_test.argmax(dim=1).cpu().numpy() ==  test_labels.cpu().numpy())*100.0


        if cfg['dataset'] == 'imagenet':
            return loss, acc_test, acc_test_v2, acc_test_s, acc_test_a, acc_test_r
        else:
            return loss, acc_test