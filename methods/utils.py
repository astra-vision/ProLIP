import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def build_cache_model(cfg, clip_model, train_loader_cache,task,proj):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    x_before_proj = clip_model.encode_image(images)
                    image_features = proj(x_before_proj)
                    image_features = F.normalize(image_features,dim=-1)
                    train_features.append(image_features)

                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        cache_dir = Path(cfg['cache_dir'])
        cache_dir.mkdir(parents=True, exist_ok=True)
        # torch.save(cache_keys, cache_dir / f'keys_task{task}.pt')
        # torch.save(cache_values, cache_dir / f'values_task{task}.pt')

    else:
        cache_dir = Path(cfg['cache_dir'])
        cache_keys = torch.load(cache_keys, cache_dir / f'keys_task{task}.pt')
        cache_values = torch.load(cache_values, cache_dir / f'values_task{task}.pt')

    return cache_keys, cache_values

def search_hp_tip(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

def get_one_hot(y_s: torch.tensor, num_classes: int):
    """
        args:
            y_s : torch.Tensor of shape [n_task, shot]
        returns
            y_s : torch.Tensor of shape [n_task, shot, num_classes]
    """
    one_hot_size = list(y_s.size()) + [num_classes]
    one_hot = torch.zeros(one_hot_size, device=y_s.device, dtype=torch.float16)
    print(one_hot)

    one_hot.scatter_(-1, y_s.unsqueeze(-1), 1)
    return one_hot

def compute_centroids_alpha(z_s: torch.tensor,
                      y_s: torch.tensor):
    """
    inputs:
        z_s : torch.Tensor of size [batch_size, s_shot, d]
        y_s : torch.Tensor of size [batch_size, s_shot]

    updates :
        centroids : torch.Tensor of size [n_task, num_class, d]
    """
    one_hot = get_one_hot(y_s, num_classes=y_s.unique().size(0))
    centroids = (one_hot*z_s/ one_hot.sum(-2, keepdim=True)).sum(1)  # [batch, K, d]
    return centroids


def compute_centroids(z_s: torch.tensor,
                      y_s: torch.tensor):
    """
    inputs:
        z_s : torch.Tensor of size [batch_size, s_shot, d]
        y_s : torch.Tensor of size [batch_size, s_shot]

    updates :
        centroids : torch.Tensor of size [n_task, num_class, d]
    """
    one_hot = get_one_hot(y_s, num_classes=y_s.unique().size(0)).transpose(1, 2) 
    # centroids = one_hot.bmm(z_s) / one_hot.sum(-1, keepdim=True)  # [batch, K, d]
    centroids = one_hot.bmm(z_s)  # [batch, K, d]
    return centroids



def compute_image_features(clip_model, loader):
    
    x_before_list, labels = [], []

    with torch.no_grad():
        for j, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            x_before_proj = clip_model.encode_image(images)
            x_before_list.append(x_before_proj)
            labels.append(target)

    x_before_proj, labels = torch.cat(x_before_list), torch.cat(labels)

    return x_before_proj, labels

def compute_image_features_test(clip_model,loader,proj,text_weights):

    acc_list = []
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            x_before_proj = clip_model.encode_image(images)

            test_features = proj(x_before_proj)
            test_features = F.normalize(test_features,dim=-1)
            logits_test = 100. * test_features @ text_weights
            acc = logits_test.argmax(dim=1).cpu() ==  target.cpu()
            acc_list += acc.detach().tolist()
        acc_test = 100 * np.mean(acc_list)
    return acc_test