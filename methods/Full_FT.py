from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import numpy as np
from tqdm import tqdm
from .method import FSCLIPmethod
from .utils import cls_acc
from .utils import compute_image_features


class FULLFT(FSCLIPmethod):

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

        print(self.cfg['train_epoch'])
        
        print('Turning off gradients for all layers except the vision encoder')

        for name, param in model.named_parameters():
            param.requires_grad = False
        for param in model.visual.parameters():
            param.requires_grad = True

        model.cuda()

        start_time = time.time()
                
        optimizer = torch.optim.Adam(model.parameters(),  lr=cfg['lr'], eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch']* len(train_loader))
        
        # Train
        print('\nStart Training procedure')
        print(cfg['lr'])
    
        for train_idx in range(self.cfg['train_epoch']):
                 
            # Train
            correct_samples, all_samples = 0, 0
            loss_list_ce = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.cfg['train_epoch']))

            for i, (images, target) in enumerate(tqdm(train_loader)):

                images, target = images.cuda(), target.cuda()
                
                x_before_proj = model.encode_image(images)
                if cfg['backbone'] == "RN50" or cfg['backbone'] == "RN101":
                    weight = model.visual.attnpool.c_proj.weight
                    bias = model.visual.attnpool.c_proj.bias
                    image_features = torch.matmul(x_before_proj,weight.T) + bias
                elif cfg['backbone'] == "ViT-B/16" or cfg['backbone'] == "ViT-B/32":
                    weight = state_dict["visual.proj"]
                    image_features = x_before_proj @ weight

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
        
        if cfg['save_checkpoints'] == True:
            save_path = Path("trained_models") / config_file /cfg['dataset'] / f"{shots}_shot" / f"{cfg['dataset']}_seed{task}.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

        # Evaluation on the test set
        print("Total time = {:.4f}".format(time.time()-start_time))        
        print('\nStart evaluation on test set')
        #########################
       
        ###evaluate on test set
        
        if cfg['dataset'] == 'imagenet':
            acc_test = self.compute_image_features_test(cfg,model, state_dict, test_loader,text_weights)
            acc_test_v2 = self.compute_image_features_test(cfg,model, state_dict, test_loader_v2,text_weights)
            acc_test_s = self.compute_image_features_test(cfg,model, state_dict, test_loader_sketch,text_weights)
            acc_test_a = self.compute_image_features_test(cfg,model, state_dict, test_loader_a,text_weights_a)
            acc_test_r = self.compute_image_features_test(cfg,model, state_dict, test_loader_r,text_weights_r)
        else:
            test_features_before_proj, test_labels = compute_image_features(model, test_loader)
            if cfg['backbone'] == "RN50" or cfg['backbone'] == "RN101":
                weight = model.visual.attnpool.c_proj.weight
                bias = model.visual.attnpool.c_proj.bias
                test_features = torch.matmul(test_features_before_proj,weight.T) + bias
            elif cfg['backbone'] == "ViT-B/16" or cfg['backbone'] == "ViT-B/32":
                weight = state_dict["visual.proj"]
                test_features = x_before_proj @ weight
            
            logits_test = 100. * test_features @ text_weights
            acc_test = np.mean(logits_test.argmax(dim=1).cpu().numpy() ==  test_labels.cpu().numpy())*100.0

        if cfg['dataset'] == 'imagenet':
            return loss, acc_test, acc_test_v2, acc_test_s, acc_test_a, acc_test_r
        else:
            return loss, acc_test



    def compute_image_features_test(self,cfg, model,state_dict, loader,text_weights):
              
        acc_list = []
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                images, target = images.cuda(), target.cuda()
                x_before_proj = model.encode_image(images)

                if cfg['backbone'] == "RN50" or cfg['backbone'] == "RN101":
                    weight = model.visual.attnpool.c_proj.weight
                    bias = model.visual.attnpool.c_proj.bias
                    test_features = torch.matmul(x_before_proj,weight.T) + bias
                elif cfg['backbone'] == "ViT-B/16" or cfg['backbone'] == "ViT-B/32":
                    weight = state_dict["visual.proj"]
                    test_features = x_before_proj @ weight

                test_features = F.normalize(test_features,dim=-1)
                logits_test = 100. * test_features @ text_weights
                acc = logits_test.argmax(dim=1).cpu() ==  target.cpu()
                acc_list += acc.detach().tolist()
            acc_test = 100 * np.mean(acc_list)
        return acc_test