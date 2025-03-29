import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import numpy as np

from tqdm import tqdm
from .method import FSCLIPmethod
from .utils import build_cache_model, search_hp_tip, cls_acc


class TIPAdapter(FSCLIPmethod):
    '''
    TIP Adapter and Tip-Adapter-F methods
    '''

    def __init__(self, args: argparse.Namespace):
        # self.normalize = args.normalize
        super().__init__(args)
        self.cfg = args
        self.lr = args['lr']
        self.epoch = args['train_epoch']
        self.shot = args['shots']
        self.init_beta = args['init_beta']
        self.init_alpha = args['init_alpha']
        self.finetune = args['finetune']

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
                # text_weights_before_a: torch.tensor,
                # text_weights_before_r: torch.tensor,
                texts: torch.tensor,
                model: nn.Module,
                state_dict,
                classnames,
                task: int,
                shots: int):
        """
        inputs:
            train_loader : torch.utils.data.DataLoader
            test_features : torch.Tensor of shape [test_data_size, 1024]
            test_labels : torch.Tensor of shape [test_data_size]
            text_weights : torch.Tensor of shape [num_shot*num_classes, 1024]
        """

        print(self.cfg['backbone'])
        cache_keys, cache_values = build_cache_model(self.cfg, model, train_loader,task)
        beta, alpha = self.cfg['init_beta'], self.cfg['init_alpha']
        
        # Feature Extraction for Validation
        print("\nExtracting visual features and labels from val set.")
        val_features, val_labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(val_loader)):
                images, target = images.cuda(), target.cuda()
                if self.cfg['backbone'] == "RN50" or self.cfg['backbone'] == "RN101":
                    _, image_features = model.encode_image(images)
                else:
                    _ , _ , image_features = model.encode_image(images)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                val_features.append(image_features)
                val_labels.append(target)
        val_features, val_labels = torch.cat(val_features), torch.cat(val_labels)

        start_time = time.time()
        
        # Enable the cached keys to be learnable
        adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(model.dtype).cuda()
        adapter.weight = nn.Parameter(cache_keys.t())
        
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=self.cfg['lr'], eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'] * len(train_loader))

        
        # Training Prodecure
        print("**** Start Training **** \n")
        best_acc, best_epoch = 0.0, 0
        for train_idx in range(self.cfg['train_epoch']):
            # Train
            adapter.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.epoch))

            for i, (images, target) in enumerate(train_loader):
                
                images, target = images.cuda(), target.cuda()
                # print("Extraction")
                with torch.no_grad():
                    if self.cfg['backbone'] == "RN50" or self.cfg['backbone'] == "RN101":
                        _, image_features = model.encode_image(images)
                    else:
                        _, _ , image_features = model.encode_image(images)

                    image_features /= image_features.norm(dim=-1, keepdim=True)
                affinity = adapter(image_features)
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * image_features @ text_weights
                tip_logits = clip_logits + cache_logits * alpha

                loss = F.cross_entropy(tip_logits, target)

                acc = cls_acc(tip_logits, target)
                correct_samples += acc / 100 * len(tip_logits)
                all_samples += len(tip_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                

            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))
        
        acc_test = self.compute_image_features_test(model, test_loader,adapter,text_weights,cache_values)
        acc_test_v2 = self.compute_image_features_test(model, test_loader_v2,adapter,text_weights,cache_values)
        acc_test_s = self.compute_image_features_test(model, test_loader_sketch,adapter,text_weights,cache_values)
        acc_test_a = self.compute_image_features_test(model, test_loader_a,adapter,text_weights_a,cache_values[:,:200])
        acc_test_r = self.compute_image_features_test(model, test_loader_r,adapter,text_weights_r,cache_values[:,:200])

        
        
        print("**** Tip-Adapter-F's test accuracy before search : {:.2f}. ****\n".format(acc))
        print("Total time = {:.4f}".format(time.time()-start_time))
       
        return loss, acc_test, acc_test_v2, acc_test_s, acc_test_a, acc_test_r

    def compute_image_features_test(self,model, loader,adapter,text_weights,cache_values):

        acc_list = []
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    if self.cfg['backbone'] == "RN50" or self.cfg['backbone'] == "RN101":
                        x_pre_proj, image_features = model.encode_image(images)
                    else:
                        _ , _ , image_features = model.encode_image(images)
                        
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                
                affinity = adapter(image_features)
                cache_logits = ((-1) * (1 - 1 * affinity)).exp() @ cache_values
                clip_logits = 100. * image_features @ text_weights
                tip_logits = clip_logits + cache_logits * 1
                acc = tip_logits.argmax(dim=1).cpu() ==  target.cpu()
                acc_list += acc.detach().tolist()

            acc_test = 100 * np.mean(acc_list)
        return acc_test