import copy
import os
from unittest.mock import CallableMixin
import warnings
import random
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm


import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
from sklearn.metrics import f1_score, average_precision_score
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, TripletMarginLoss
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from pytorchvideo.transforms import (
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
)

from dataloader import omniDataLoader, default_collate, val_collate
from model import build_model
import pickle


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    acc = acc.squeeze()
    result = np.mean(np.mean(acc - np.diag(np.diag(acc)), 1))
    diag_mean = np.mean(np.diag(np.diag(acc)))
    print(f'non_diag_mean: {result}', flush=True)
    print(f'diag_mean: {diag_mean}', flush=True)
    return result


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def train_epoch(epoch, data_loader, model, optimizer, ema_optimizer, criterion, writer, use_cuda, flag, args, action_flag=False):
    print('train at epoch {}'.format(epoch), flush=True)
    count = 0
    losses = []
    supervised_sub_losses = []
    supervised_act_losses = []
    ss_contrastive_losses = []
    sa_contrastive_losses = []
    ortho_sub_losses = []
    ortho_act_losses = []
    act_acc = []
    sub_acc = []

    model.train()

    if flag:
        for i, (clips, sv_clips, sa_clips, targets, actions, _) in enumerate(tqdm(data_loader)):
            assert len(clips) == len(targets)
    
            if use_cuda:
                clips = Variable(clips.type(torch.FloatTensor)).cuda()
                sv_clips = Variable(sv_clips.type(torch.FloatTensor)).cuda()
                sa_clips = Variable(sa_clips.type(torch.FloatTensor)).cuda()
                targets = Variable(targets.type(torch.LongTensor)).cuda()
                actions = Variable(actions.type(torch.LongTensor)).cuda()
            else:
                clips = Variable(clips.type(torch.FloatTensor))
                sv_clips = Variable(sv_clips.type(torch.FloatTensor))
                sa_clips = Variable(sa_clips.type(torch.FloatTensor))
                targets = Variable(targets.type(torch.LongTensor))
                actions = Variable(actions.type(torch.LongTensor))
    
            optimizer.zero_grad()
            
            output_subjects, output_actions, features, act_features, bsubq, bactq = model(clips)
            _, _, sv_features, sv_act_features, _, _ = model(sv_clips)
            _, _, sa_features, sa_act_features, _, _ = model(sa_clips)
            
            subqloss = 0 
            actqloss = 0    
                           
            for subq in bsubq:
                dist = abs(cosine_pairwise_dist(subq, subq))
                subqloss += torch.sum(dist - torch.eye(subq.shape[0]).cuda())
            
            for actq in bactq:
                dist = abs(cosine_pairwise_dist(actq, actq))
                actqloss += torch.sum(dist - torch.eye(actq.shape[0]).cuda())
                

            sv_contrastive_loss = nn.TripletMarginLoss()(features, sv_features, sa_features)
            sa_contrastive_loss = nn.TripletMarginLoss()(act_features, sa_act_features, sv_act_features)
            
            sub_loss = criterion(output_subjects, targets)
            act_loss = criterion(output_actions, actions)
            
            output_subjects = torch.argmax(output_subjects, dim=1)
            acc = torch.sum(output_subjects == targets)
            sub_acc.append(acc)
            
            output_actions = torch.argmax(output_actions, dim=1)
            acc = torch.sum(output_actions == actions)
            act_acc.append(acc)
           
            loss = sub_loss + act_loss + sv_contrastive_loss + sa_contrastive_loss + subqloss + actqloss
            
            if 3 < i < 5:
                act = torch.stack([acc for acc in act_acc])
                act_acc_pred = torch.sum(act) / (len(act) * args.batch_size)
                sub = torch.stack([acc for acc in sub_acc])
                sub_acc_pred = torch.sum(sub) / (len(sub) * args.batch_size)
                # print(act_acc_pred, sub_acc_pred)
                print(f'pred sub: {output_subjects}, GT: {targets}, pred act: {output_actions}, GT: {actions}, features: {features.shape}', flush=True)
                    
            
            supervised_sub_losses.append(sub_loss.item())
            supervised_act_losses.append(act_loss.item())
            ss_contrastive_losses.append(sv_contrastive_loss.item())
            sa_contrastive_losses.append(sa_contrastive_loss.item())
            ortho_sub_losses.append(subqloss.item())
            ortho_act_losses.append(actqloss.item())
            
           
            losses.append(loss.item())
            loss = loss / accumulation_steps
            loss.backward()             
            
            if (i+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema_optimizer.step()
                
            losses.append(loss.item())
    
            del sub_loss, act_loss, loss, sa_contrastive_loss, sv_contrastive_loss, actqloss, subqloss, output_subjects, output_actions, features, act_features, sv_features, sv_act_features, sa_features, sa_act_features, clips, sv_clips, sa_clips, targets, actions
               
        act = torch.stack([acc for acc in act_acc])
        act_acc = torch.sum(act) / (len(act) * args.batch_size)
        sub = torch.stack([acc for acc in sub_acc])
        sub_acc = torch.sum(sub) / (len(sub) * args.batch_size)
            
    print('Training Epoch: %d, Loss: %.4f, SL: %.4f, AL: %.4f, SCL: %.4f, ACL: %.4f, OSL: %.4f, OAL: %.4f' % (epoch, np.mean(losses), np.mean(supervised_sub_losses),  np.mean(supervised_act_losses), np.mean(ss_contrastive_losses), np.mean(sa_contrastive_losses), np.mean(ortho_sub_losses), np.mean(ortho_act_losses)), flush=True)
    print('Training Epoch: %d, View Accuracy: %.4f' % (epoch, sub_acc), flush=True)
    
    print('Training Epoch: %d, Action Accuracy: %.4f' % (epoch, act_acc), flush=True)
        
            
    writer.add_scalar('Training Loss', np.mean(losses), epoch)
    writer.add_scalar('View Loss', np.mean(supervised_sub_losses), epoch)
    writer.add_scalar('Action Loss', np.mean(supervised_act_losses), epoch)
    writer.add_scalar('View Contrastive Loss', np.mean(ss_contrastive_losses), epoch)
    writer.add_scalar('Action Contrastive Loss', np.mean(sa_contrastive_losses), epoch)
      
    return model
      


def val_epoch(cfg, epoch, data_loader, model, writer, use_cuda, args, action_flag=False):
    print('validation at epoch {}'.format(epoch))
    model.eval()

    results = {}
    act_acc = []
    sub_acc = []
    for i, (clips, labels, action_targets, keys) in enumerate(tqdm(data_loader)):
        clips = Variable(clips.type(torch.FloatTensor))
        labels =  Variable(labels.type(torch.FloatTensor))
        action_targets =  Variable(action_targets.type(torch.FloatTensor))        
        
        assert len(clips) == len(labels)
                        
        
        with torch.no_grad():
            if use_cuda:
                clips = clips.cuda()
                labels = labels.cuda()
                action_targets = action_targets.cuda()
      
            output_subjects, output_actions, features, act_features, _, _ = model(clips)
            output_subjects = torch.argmax(output_subjects, dim=1)
            output_actions = torch.argmax(output_actions, dim=1)
            acc = torch.sum(output_actions == action_targets)
            act_acc.append(acc)
            if i == 3:
                # print(output_actions, action_targets, flush=True)
                # print(output_subjects, labels, flush=True)
                act_pred = torch.stack([acc for acc in act_acc])
                act_acc_pred = torch.sum(act_pred) / (len(act_pred) * args.batch_size)
                # print(act_acc_pred)
                sub_pred = torch.stack([acc for acc in sub_acc])                
                sub_acc_pred = torch.sum(sub_pred) / (len(sub_pred) * args.batch_size)
                # print(sub_acc_pred)
            
            acc = torch.sum(output_subjects == labels)
            sub_acc.append(acc)
    
    sub = torch.stack([acc for acc in sub_acc])                
    sub_acc = torch.sum(sub) / (len(sub) * args.batch_size)
    act = torch.stack([acc for acc in act_acc])
    act_acc = torch.sum(act) / (len(act) * args.batch_size)
    print('Validation Epoch: %d, Action Accuracy: %.4f' % (epoch, act_acc), flush=True)
    print('Validation Epoch: %d, View Accuracy: %.4f' % (epoch, sub_acc), flush=True)
    return act_acc
        
    

def train_model(cfg, run_id, save_dir, use_cuda, args, writer):
    shuffle = True
    print("Run ID : " + args.run_id)
   
    print("Parameters used : ")
    print("batch_size: " + str(args.batch_size))
    print("lr: " + str(args.learning_rate))

    transform_train = Compose(
        [
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            RandomShortSideScale(
                min_size=224,
                max_size=256,
            ),
            RandomCrop(args.input_dim),
            RandomHorizontalFlip(p=0.5)
            
        ]
    )
    transform_test = Compose(
        [
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ShortSideScale(
                size=256
            ),
            CenterCrop(args.input_dim)
        ]
    )
    
    flag = True if args.model_version == 'v3'] else False    
    train_data_gen = omniDataLoader(cfg, 'train', transform=transform_train, flag=flag)
    val_data_gen = omniDataLoader(cfg, 'test', 1.0, args.num_frames, skip=skip, transform=transform_test, flag=False)
    
    train_dataloader = DataLoader(train_data_gen, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=True, collate_fn=default_collate)
    val_dataloader = DataLoader(val_data_gen, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=False, collate_fn=val_collate)
    
    print("Number of training samples : " + str(len(train_data_gen)))
    print("Number of testing samples : " + str(len(val_data_gen)))
    
    steps_per_epoch = len(train_data_gen) / args.batch_size
    print("Steps per epoch: " + str(steps_per_epoch))

    num_views = 3 
    model = build_model(args.model_version, num_views, cfg.num_actions)
    
    #####################################################################################################################
    num_gpus = len(args.gpu.split(','))
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    #####################################################################################################################
    
    if args.checkpoint:
        pretrained_weights = torch.load(args.checkpoint)['state_dict']
        model.load_state_dict(pretrained_weights, strict=True)
        print("loaded", flush=True)

    if args.optimizer == 'ADAM':
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'ADAMW':
        print("Using ADAMW optimizer")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        
    ema_model = copy.deepcopy(model)
    ema_optimizer = WeightEMA(model, ema_model, args.learning_rate, alpha=args.ema_decay)
    
    criterion = CrossEntropyLoss()
    
    max_fmap_score, fmap_score = -1, -1
    # loop for each epoch
    for epoch in range(args.num_epochs):
        model = train_epoch(epoch, train_dataloader, model, optimizer, ema_optimizer, criterion, writer, use_cuda, flag, args, action_flag=False)
        if epoch % args.validation_interval == 0:
            score1 = val_epoch(cfg, epoch, val_dataloader, model, None, use_cuda, args)
            fmap_score = score1
            if flag:
                score2 = val_epoch(cfg, epoch, val_dataloader, ema_model, writer, use_cuda, args)
                fmap_score = max(score1, score2)
         
        #if fmap_score > max_fmap_score:
        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))
        save_file_path = os.path.join(save_dir, 'model_{}_{:.4f}.pth'.format(epoch, fmap_score))
        if flag:
            save_model = model if score1 > score2 else ema_model
        else:
            save_model = model
        states = {
            'epoch': epoch + 1,
            'state_dict': save_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
        max_fmap_score = fmap_score
                
                
def cosine_pairwise_dist(x, y):
    assert x.shape[1] == y.shape[1], "both sets of features must have same shape"
    return nn.functional.cosine_similarity(x[:, None, :], y[None, :, :], dim=-1)    
