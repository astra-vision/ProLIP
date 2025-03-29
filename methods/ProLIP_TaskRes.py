from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import argparse
import numpy as np
from .method import FSCLIPmethod
from .utils import cls_acc
from .utils import compute_image_features
from copy import deepcopy


class VisProjRN_TaskRes(nn.Module):
    def __init__(self, weight, bias,text_weights):
        super(VisProjRN_TaskRes, self).__init__()
        output_dim, input_dim = weight.size()

        self.linear = nn.Linear(output_dim, input_dim)
        
        self.linear.weight = nn.Parameter(weight)
        self.linear.bias = nn.Parameter(bias)

        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear1.weight = nn.Parameter(deepcopy(weight))
        self.linear1.bias = nn.Parameter(deepcopy(bias))

        ### ProLIP frozen projection layer 
        self.linear1.weight.requires_grad =  False
        self.linear1.bias.requires_grad = False

        ### ProLIP visual projection matrix to train        
        self.linear.weight.requires_grad =  True
        self.linear.bias.requires_grad = False
    
        self.text_weights = text_weights
        
        ### Residual in TaskRes
        self.R = nn.Parameter(torch.zeros_like(self.text_weights))

    def forward(self, x):
        return 100. * F.normalize(self.linear(x),dim=-1) @ self.text_weights + 0.1 *  100. * F.normalize(self.linear1(x),dim=-1) @ self.R


class ProLIP_TaskRes(FSCLIPmethod):

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
        weight_init =  model.visual.attnpool.c_proj.weight
        bias_init = model.visual.attnpool.c_proj.bias
        weight_init_copy = copy.deepcopy(weight_init)

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

        lr_v = cfg["lr_v"]
        if cfg['lambda_funct_1_N']:
            lambda_v = 1/shots
        elif cfg['lambda_funct_1_N2']:
            lambda_v = 1/shots**2
        else:
            lambda_v = cfg['lambda_v']

        print(lr_v)
        print(lambda_v)
        
        ###reinitialize proj

        proj = VisProjRN_TaskRes(weight_init,bias_init,text_weights)
        start_time = time.time()

        optimizer = torch.optim.Adam(proj.parameters(),  lr=lr_v, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'])

        # Train
        print('\nStart Training procedure')
            
        cnt = 0
        for train_idx in range(self.cfg['train_epoch']):
                 
            # Train
            correct_samples, all_samples = 0, 0
            loss_list_ce = []
            loss_list_mse = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.cfg['train_epoch']))

            if (cnt+1)% self.cfg['aug_views'] == 0:
                cnt = 0
            else:
                cnt +=1

            train_x_before = train_x_before_list[cnt]
            
            train_x_before , target = train_x_before.cuda(), train_labels.cuda()
    
            logits = proj(train_x_before)

            initial_params = weight_init_copy.view(-1)
            fine_tuned_params = proj.linear.weight.view(-1)
            
            mse_loss = mse(initial_params,fine_tuned_params)
           
            loss1 = F.cross_entropy(logits, target)
            loss2 =  mse_loss

            loss = loss1 + lambda_v * loss2

            acc = cls_acc(logits, target)
            correct_samples += acc / 100 * len(logits)
            all_samples += len(logits)
            loss_list_ce.append(loss1.item())
            loss_list_mse.append(loss2.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
           
            print('Acc: {:.4f} ({:}/{:}), Loss_ce: {:.4f}, Loss_mse: {:.4f}'.format(correct_samples / all_samples, correct_samples, all_samples, sum(loss_list_ce)/len(loss_list_ce), sum(loss_list_mse)/len(loss_list_mse)))

        torch.cuda.empty_cache()
        train_x_before = train_x_before.cpu()
        train_labels = train_labels.cpu()

        del train_x_before
        del train_labels
        
        if cfg['save_checkpoints'] == True:
            save_path = Path("trained_models") / config_file /cfg['dataset'] / f"{shots}_shot" / f"{cfg['dataset']}_seed{task}.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(proj.state_dict(), save_path)

    
        # Evaluation on the test set
        print("Total time = {:.4f}".format(time.time()-start_time))        
        print('\nStart evaluation on test set')
        #########################
        
        if cfg['dataset'] == 'imagenet':

            acc_test = self.compute_image_features_test(model, test_loader,proj,text_weights)
            acc_test_v2, acc_test_s, acc_test_a, acc_test_r = None, None, None, None

        else:
            test_features_before, test_labels = compute_image_features(model, test_loader)
            logits_test = proj(test_features_before)
            acc_test = np.mean(logits_test.argmax(dim=1).cpu().numpy() ==  test_labels.cpu().numpy())*100.0


        if cfg['dataset'] == 'imagenet':
            return loss, acc_test, acc_test_v2, acc_test_s, acc_test_a, acc_test_r
        else:
            return loss, acc_test

    def compute_image_features_test(self,model, loader,proj,text_weights):

        acc_list = []
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    x_before_proj = model.encode_image(images)

                logits = proj(x_before_proj)
                
                acc = logits.argmax(dim=1).cpu() ==  target.cpu()
                acc_list += acc.detach().tolist()

            acc_test = 100 * np.mean(acc_list)
        return acc_test