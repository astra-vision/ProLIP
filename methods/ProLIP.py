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
from .utils import compute_image_features, compute_image_features_test


class VisProjRN(nn.Module):
    def __init__(self, weight, bias):
        super(VisProjRN, self).__init__()
        output_dim, input_dim = weight.size()

        self.linear = nn.Linear(output_dim, input_dim)
        
        self.linear.weight = nn.Parameter(weight)
        self.linear.bias = nn.Parameter(bias)
        
        self.linear.weight.requires_grad =  True
        self.linear.bias.requires_grad = False
    
    def forward(self, x_before_proj):
        return self.linear(x_before_proj)

class VisProjViT(nn.Module):
    def __init__(self, vit_proj):
        super(VisProjViT, self).__init__()
       
        self.vit_proj = nn.Parameter(vit_proj)
        self.vit_proj.requires_grad = True

    def forward(self,x_before_proj):

        x = x_before_proj @ self.vit_proj
        return x


class ProLIP(FSCLIPmethod):

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
            weight_init_copy = copy.deepcopy(weight_init)

        elif cfg['backbone'] == "ViT-B/16" or cfg['backbone'] == "ViT-B/32":
            vit_proj = state_dict["visual.proj"]
            vit_proj_copy = copy.deepcopy(vit_proj)


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
        
        
        if cfg['search_lr']:
            best_acc = 0.0
            print("**** Searching for best lr **** \n")
            lr_list = [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
            lambda_list = [10,1,0.1,0.01,0.001,0.0001,0]
            for init_lr in lr_list:
                for init_lambda in lambda_list:
                    proj = self.search_init_hp(mse,cfg,init_lr,init_lambda, model,state_dict,text_weights,train_x_before_list, train_labels)

                    val_features_before, val_labels = compute_image_features(model, val_loader)
                    val_features = proj(val_features_before)
                    val_features = F.normalize(val_features,dim=-1)
                    logits = 100. * val_features @ text_weights

                    acc = cls_acc(logits, val_labels)
                    print(init_lr)
                    print(init_lambda)
                    print(acc)
                    if acc > best_acc:
                        best_acc = acc
                        lr_v = init_lr
                        lambda_v = init_lambda
                    
        else:
            lr_v = cfg["lr_v"]
            if cfg['lambda_funct_1_N']:
                lambda_v = 1/shots
            elif cfg['lambda_funct_1_N2']:
                lambda_v = 1/shots**2
            else:
                lambda_v = cfg['lambda_v']

        print(lr_v)
        print(lambda_v)

        if cfg['search_lr']:
            save_path = Path("results_lr") / config_file / f"{cfg['dataset']}{shots}_shot_lr.txt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("a", encoding="utf-8") as file:
                file.write(f"{lr_v}, {lambda_v}\n")

        ###reinitialize proj

        if cfg['backbone'] == "RN50" or cfg['backbone'] == "RN101":
            proj = VisProjRN(weight_init,bias_init)
        elif cfg['backbone'] == "ViT-B/16" or cfg['backbone'] == "ViT-B/32":
            proj = VisProjViT(vit_proj)

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
    
            image_features = proj(train_x_before)
            image_features = F.normalize(image_features,dim=-1)

            logits = 100. * image_features @ text_weights

            if cfg['backbone'] == "RN50" or cfg['backbone'] == "RN101":
                initial_params = weight_init_copy.view(-1)
                fine_tuned_params = proj.linear.weight.view(-1)

            elif cfg['backbone'] == "ViT-B/16" or cfg['backbone'] == "ViT-B/32":
                initial_params = vit_proj_copy.view(-1)
                fine_tuned_params = proj.vit_proj.view(-1)

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
        
        if cfg['dataset'] == 'imagenet':
            acc_test = compute_image_features_test(model, test_loader,proj,text_weights)
            if cfg["SUBSAMPLE_CLASSES"] == "all":
                acc_test_v2 = compute_image_features_test(model, test_loader_v2,proj,text_weights)
                acc_test_s = compute_image_features_test(model, test_loader_sketch,proj,text_weights)
                acc_test_a = compute_image_features_test(model, test_loader_a,proj,text_weights_a)
                acc_test_r = compute_image_features_test(model, test_loader_r,proj,text_weights_r)
            else:
                acc_test_v2,acc_test_s, acc_test_a, acc_test_r = 0, 0, 0, 0
        
        else:
  
            test_features_before, test_labels = compute_image_features(model, test_loader)
            test_features = proj(test_features_before)
            test_features = F.normalize(test_features,dim=-1)
            logits_test = 100. * test_features @ text_weights

            acc_test = np.mean(logits_test.argmax(dim=1).cpu().numpy() ==  test_labels.cpu().numpy())*100.0


        if cfg['dataset'] == 'imagenet':
            return loss, acc_test, acc_test_v2, acc_test_s, acc_test_a, acc_test_r
        else:
            return loss, acc_test

      
    def search_init_hp(self,mse, cfg, lr_v,lambda_v, model, state_dict, text_weights,train_x_before_list,train_labels):

        if cfg['backbone'] == "RN50" or cfg['backbone'] == "RN101":
            weight_init =  model.visual.attnpool.c_proj.weight
            bias_init = model.visual.attnpool.c_proj.bias
            weight_init_copy = copy.deepcopy(weight_init)
            proj = VisProjRN(weight_init_copy,bias_init)

        elif cfg['backbone'] == "ViT-B/16" or cfg['backbone'] == "ViT-B/32":
            vit_proj = state_dict["visual.proj"]            
            vit_proj_copy = copy.deepcopy(vit_proj)
            proj = VisProjViT(vit_proj_copy)
            

        optimizer = torch.optim.Adam(proj.parameters(), lr=lr_v, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'])

        # Train
        print('\nStart Training procedure')

        cnt = 0

        for train_idx in range(self.cfg['train_epoch']):
            # Train
            if train_idx == self.cfg['train_epoch']-1:
                print('Train Epoch: {:} / {:}'.format(train_idx, self.cfg['train_epoch']))

            if (cnt+1)% cfg['aug_views']==0:
                cnt=0
            else:
                cnt+=1
            
            train_x_before = train_x_before_list[cnt]
            train_x_before , target = train_x_before.cuda(), train_labels.cuda()

            image_features = proj(train_x_before)
            image_features = F.normalize(image_features,dim=-1)

            logits = 100. * image_features @ text_weights

            if cfg['backbone'] == "RN50" or cfg['backbone'] == "RN101":
                initial_params = weight_init.view(-1)
                fine_tuned_params = proj.linear.weight.view(-1)

            elif cfg['backbone'] == "ViT-B/16" or cfg['backbone'] == "ViT-B/32":
                initial_params = vit_proj.view(-1)
                fine_tuned_params = proj.vit_proj.view(-1)

            mse_loss = mse(initial_params,fine_tuned_params)
            loss1 = F.cross_entropy(logits, target)
            loss2 =  mse_loss

            loss = loss1 + lambda_v * loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        return proj