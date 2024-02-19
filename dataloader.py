from PIL import Image
import csv
import os
import sys
import random
import pandas as pd
import cv2
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from configuration import build_config
from torch.autograd.variable import Variable
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
    Resize
    
)
from pytorchvideo.transforms import (
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)

from model import build_model


def default_collate(batch):
    count = 0
    anchor_frames, sa_frames, ss_frames, targets, actions, keys = [], [], [], [], [], []
    for item in batch:
        #print(item[0].shape, item[1].shape, item[2].shape, item[3], item[4], item[5], flush=True)
        if item[0].shape == torch.Size([16, 3, 224, 224]) and item[1].shape == torch.Size([16, 3, 224, 224]) and item[2].shape == torch.Size([16, 3, 224, 224]) and item[3] is not None and item[4] is not None:
            anchor_frames.append(item[0])
            ss_frames.append(item[1])
            sa_frames.append(item[2])
            targets.append(item[3])
            actions.append(item[4])
            keys.append(item[5])
    #print(len(anchor_frames))
    anchor_frames = torch.stack(anchor_frames)
    sa_frames = torch.stack(sa_frames)
    ss_frames = torch.stack(ss_frames)
    targets = torch.tensor(targets)
    actions = torch.tensor(actions)
    return anchor_frames, ss_frames, sa_frames, targets, actions, keys
    
    
def val_collate(batch):
    count = 0
    anchor_frames, targets, actions, keys = [], [], [], []
    for item in batch:
        if item[0].shape == torch.Size([16, 3, 224, 224]) and item[1] is not None and item[2] is not None:
            anchor_frames.append(item[0])
            targets.append(item[1])
            actions.append(item[2])
            keys.append(item[3])
    anchor_frames = torch.stack(anchor_frames)
    targets = torch.tensor(targets)
    actions = torch.tensor(actions)

    return anchor_frames, targets, actions, keys
        
        
class omniDataLoader(Dataset):
    def __init__(self, cfg, data_split, height=270, width=480, shuffle=True, transform=None, flag=False):
        self.dataset = cfg.dataset
        self.flag = flag
        self.data_split = data_split
        self.videos_folder = cfg.videos_folder
        if data_split == "train":
           self.annotations = cfg.train_annotations
        else:
           self.annotations = cfg.test_annotations
        df = pd.read_csv(self.annotations)
        self.videos = []
        self.data = {}
        self.actions = []
        self.views = []
        if self.dataset != 'ntu_rgbd_60':
            hdf5_list = os.listdir(f'/home/siddiqui/Action_Biometrics-RGB/frame_data/{self.dataset}/')
        else:
            hdf5_list = os.listdir(f'/home/siddiqui/Action_Biometrics-RGB/frame_data/ntu_rgbd_120/')
        for count, row in enumerate(open(self.annotations, 'r').readlines()[1:]):
            if self.dataset != "numa":
                video_id, subject, action, placeholder1, placeholder2, placeholder3 = row.split(',')
            else:
                video_id, subject, action, viewpoint = row.split(',')       
                    
            if self.dataset == 'ntu_rgbd_120' or self.dataset == 'ntu_rgbd_60':
                if f'{video_id}.hdf5' in hdf5_list:
                    if df['subject'].value_counts()[int(subject)] < 2:
                        print(row, flush=True)
                        continue
                    self.videos.append([video_id, subject, action, placeholder1, placeholder2, placeholder3]) 
                    view = ';'.join([placeholder1, placeholder3])
                    if action not in self.actions:
                        self.actions.append(action)
                    if view not in self.views:
                        self.views.append(view)
                    if f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}_{view}" not in self.data:
                        self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}_{view}"] = []
                    self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}_{view}"].append([subject, action, video_id, placeholder1, placeholder2, placeholder3, view])
                    
            elif self.dataset == 'pkummd':
                if int(placeholder1) > int(placeholder2):
                    placeholder1, placeholder2 = placeholder2, placeholder1
                if f'{video_id}_{int(subject)-1}_{action}_{placeholder1}_{placeholder2}.hdf5' in hdf5_list or f'{video_id}_{int(subject)+8}_{action}_{placeholder1}_{placeholder2}.hdf5' in hdf5_list:
                    if df['id'].value_counts()[int(subject)] < 2:
                        print('not enough samples: {row}', flush=True)
                        continue
                    self.videos.append([video_id, subject, action, placeholder1, placeholder2, placeholder3])
                    if action not in self.actions:
                        self.actions.append(action)
                    if f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}" not in self.data:
                        self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"] = []
                    self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"].append([subject, action, video_id, placeholder1, placeholder2, placeholder3])
                    
            elif self.dataset == 'numa':
                if f'{video_id[:-4]};{action}.hdf5' in hdf5_list:
                    if df['subject'].value_counts()[int(subject)] < 2:
                        print(row, flush=True)
                        continue
                    self.videos.append([video_id, subject, action, viewpoint])
                    if action not in self.actions:
                        self.actions.append(action)
                    if f"{subject}_{action}_{video_id}_{viewpoint}" not in self.data:
                        self.data[f"{subject}_{action}_{video_id}_{viewpoint}"] = []
                    self.data[f"{subject}_{action}_{video_id}_{viewpoint}"].append([subject, action, video_id, viewpoint])
                else:
                    print('video not found: {video_id}')

        if shuffle and data_split == 'train':
            random.shuffle(self.videos)

        self.actions = sorted(self.actions)
        self.views = sorted(self.views)
        #print(len(self.subjects), len(self.actions), len(self.videos), len(self.views), self.views, flush=True)
        self.height = height
        self.width = width
        self.transform = transform
        self.num_frames = 16

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        if self.dataset == 'ntu_rgbd_120' or self.dataset == 'ntu_rgbd_60':
            if self.flag:
                anchor = self.videos[index]
                video_id, sub, act, camera, rep, setup = anchor[0], anchor[1], anchor[2], anchor[3], anchor[4], anchor[5]
                row = [sub, act, video_id, camera, rep, setup]
                
                sv = random.choice([sameview for sameview in self.data.keys() if sameview.split("_")[4] == camera and sameview.split("_")[1] != act])
                sa = random.choice([sameaction for sameaction in self.data.keys() if sameaction.split("_")[4] != camera and sameaction.split("_")[1] == act])
                
                anchor_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[sa])
                sa_sub, sa_act, sa_video_id, sa_start_frame, sa_end_frame, sa_scene = row[0], row[1], row[2], row[3], row[4], row[5]
                sa_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[sv])
                sv_sub, sv_act, sv_video_id, sv_start_frame, sv_end_frame =  row[0], row[1], row[2], row[3], row[4]
                sv_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
              
                action = self.actions.index(act)
                camera = int(camera) - 1
                return anchor_frames, sv_frames, sa_frames, camera, action, '_'.join([video_id[8:12], video_id[0:4], video_id[4:8], video_id[12:16], video_id[16:20]])
              
            else:                
                row = self.videos[index]
                video_id, subject, action = row[0], row[1], row[2]
                camera, rep, setup = row[3], row[4], row[5]
                row = [subject, action, video_id, camera, rep, setup]
                view = ';'.join([camera, setup])
                frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                action = self.actions.index(action)
                camera = int(camera) - 1   
                return frames, camera, action, '_'.join([video_id[8:12], video_id[0:4], video_id[4:8], video_id[12:16], video_id[16:20]])
            
            
        elif self.dataset == 'pkummd':
            camera_index = ['L', 'M', 'R']
            if self.flag:
                anchor = self.videos[index]
                video_id, sub, act, start_frame, end_frame = anchor[0], anchor[1], anchor[2], anchor[3], anchor[4]
                camera = video_id[-1]
                row = [sub, act, video_id, start_frame, end_frame]
                
                sv = random.choice([sameview for sameview in self.data.keys() if sameview.split("_")[2][-1] == camera and sameview.split("_")[1] != act])
                sa = random.choice([sameaction for sameaction in self.data.keys() if sameaction.split("_")[2][-1] != camera and sameaction.split("_")[1] == act])       
                
                anchor_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[sa])
                sa_sub, sa_act, sa_video_id, sa_start_frame, sa_end_frame, sa_scene = row[0], row[1], row[2], row[3], row[4], row[5]
                sa_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[sv])
                sv_sub, sv_act, sv_video_id, sv_start_frame, sv_end_frame =  row[0], row[1], row[2], row[3], row[4]
                sv_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
              
                camera = camera_index.index(camera)
                action = self.actions.index(act)
                return anchor_frames, sv_frames, sa_frames, camera, action, '_'.join([str(sub), video_id, str(action), str(start_frame), str(end_frame), video_id[-1]])
                
            else:
                row = self.videos[index]
                video_id, subject, action, start_frame, end_frame = row[0], row[1], row[2], int(row[3]), int(row[4])
                camera = video_id[-1]
                row = [subject, action, video_id, start_frame, end_frame]
                frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                action = self.actions.index(action)       
                camera = camera_index.index(camera)
                return frames, camera, action, '_'.join([str(subject), video_id, str(action), str(start_frame), str(end_frame), video_id[-1]])
           
        elif self.dataset == 'numa':
            if self.flag:
                anchor = self.videos[index]
                video_id, sub, act, viewpoint = anchor[0], anchor[1], anchor[2], anchor[3]
                row = [sub, act, video_id, viewpoint]
                print(row)
                sv = random.choice([sameview for sameview in self.data.keys() if sameview.split("_")[-1] == viewpoint and sameview.split("_")[1] != act])
                sa = random.choice([sameaction for sameaction in self.data.keys() if sameaction.split("_")[-1] != viewpoint and sameaction.split("_")[1] == act])
                
                anchor_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                row = random.choice(self.data[sa])
                print(row)
                sa_sub, sa_act, sa_video_id, sa_viewpoint = row[0], row[1], row[2], row[3]
                sa_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[sv])
                print(row)
                sv_sub, sv_act, sv_video_id, sv_viewpoint =  row[0], row[1], row[2], row[3]
                sv_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
              
                action = self.actions.index(act)
                print(act, action, self.actions)
                viewpoint = int(viewpoint) - 1 
                return anchor_frames, sv_frames, sa_frames, viewpoint, action, '_'.join([str(sub), video_id, str(action), str(viewpoint), video_id[9:11]])
              
            else:
                row = self.videos[index]
                video_id, subject, action, viewpoint = row[0], row[1], row[2], row[3]
                row = [subject, action, video_id, viewpoint]
                frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                action = self.actions.index(action)
                viewpoint = int(viewpoint) - 1     
                return frames, viewpoint, action, '_'.join([str(subject), video_id, str(action), str(viewpoint), video_id[9:11]])
            
                
            
def frame_creation(row, dataset, videos_folder, height, width, num_frames, transform):    
    if dataset == "ntu_rgbd_120" or dataset == 'ntu_rgbd_60':
        list16 = []
        subject, action, video_id, start_frame, end_frame = row[0], row[1], row[2], row[3], row[4]
        frames = h5py.File(os.path.join(videos_folder, f'{video_id}.hdf5'), 'r')
        frames = frames['default'][:]
        frames = torch.from_numpy(frames).float()
        
        frame_indexer = np.linspace(0, int(frames.shape[0]) - 1 , num_frames).astype(int)
        for i, frame in enumerate(frames):
            if i in frame_indexer:
                list16.append(frame)
        frames = torch.stack([frame for frame in list16])
        
        for i, frame in enumerate(frames):
            frames[i] = frames[i] / 255.
            
        if transform:
            frames = frames.transpose(0, 1)
            frames = transform(frames)
            frames = frames.transpose(0, 1)
        return frames
        
        
    elif dataset == "pkummd":
        skeleton = True
        list32 = []
        subject, action, video_id, start_frame, end_frame = row[0], row[1], row[2], int(row[3]), int(row[4])
        
        if not skeleton:
            try:
                frames = h5py.File(os.path.join(videos_folder, f'{video_id}_{int(subject)-1}_{action}_{start_frame}_{end_frame}.hdf5'), 'r')
            except OSError:
                frames = h5py.File(os.path.join(videos_folder, f'{video_id}_{int(subject)+8}_{action}_{start_frame}_{end_frame}.hdf5'), 'r')
            frames = frames['default'][:]
            frames = torch.from_numpy(frames).float()
            
            frame_indexer = np.linspace(start_frame, end_frame-1, 16).astype(int)
            for i, frame in enumerate(frames, start_frame):
                if i in frame_indexer:
                    list32.append(frame)
            frames = torch.stack([frame for frame in list32])        
            
            for i, frame in enumerate(frames):
                frames[i] = frames[i] / 255.
                
            if transform:
                frames = frames.transpose(0, 1)
                frames = transform(frames)
                frames = frames.transpose(0, 1)
            
            return frames
            
            
        else:
            path = '/squash/PKUMMDs/PKUMMD/SKELETON_VIDEO/'
            processed_action_skeletons = []     
            frame_indexer = np.linspace(start_frame, end_frame-1, 16).astype(int)
            
            for i in sorted(frame_indexer):
                action_skeletons = open(os.path.join(path, f'{video_id}.txt'), 'r').readlines()[i].split(' ')
                frame_skeleton = [float(ele) for ele in action_skeletons]
                frame_skeleton = torch.tensor(frame_skeleton)
                
                sub1 = frame_skeleton[:75].reshape((3, 25)) # first 75 entries are 3x25 skeletons for subject 1
                sub2 = frame_skeleton[75:].reshape((3, 25)) # last 75 entries are 3x25 skeletons for subject 2 (0s if no sub 2)
                frame_skeleton = torch.stack((sub1, sub2), dim=2) # stack them to get same 3x25x2 shape as PKUMMD website
                processed_action_skeletons.append(frame_skeleton)
                
            processed_skeletons = torch.stack([ele for ele in processed_action_skeletons]) # 16x3x25x2: skeletons from 16 equidistant frames in action range
            return processed_skeletons

    elif dataset == "numa":
        list16 = []
        subject, action, video_id, viewpoint = row
        with h5py.File(f'{videos_folder}/{video_id[:-4]};{action}.hdf5', 'r') as f:
            frames = f['default'][:]
            frames = torch.from_numpy(frames)
            frames = frames.type(torch.float32)
            
        frame_indexer = np.linspace(0, len(frames)-1, num_frames).astype(int)
        
        for i in frame_indexer:
            list16.append(frames[i])
        frames = torch.stack([frame for frame in list16])
            
        for i, frame in enumerate(frames):
            frames[i] = frames[i] / 255.
            
        if transform:
            frames = frames.transpose(0, 1)
            frames = transform(frames)
            frames = frames.transpose(0, 1)
        return frames
        
        
if __name__ == '__main__':
    shuffle = False
    cfg = build_config('pkummd')
    transform_train = Compose(
                [
                    Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
                    RandomShortSideScale(
                        min_size=224,
                        max_size=256,
                    ),
                    RandomCrop(224),
                    RandomHorizontalFlip(p=0.5)
                ]
            )
    transform_test = Compose(
                [
                    Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
                    ShortSideScale(
                        size=256
                    ),
                    CenterCrop(224)
                ]
            )
   
    for (clips, sv_clips, sa_clips, views, actions, keys) in tqdm(dataloader):
        print(clips.shape, sv_clips.shape, sa_clips.shape, views, actions)
        print(actions.shape, views.shape)
        exit()
    

        
