from PIL import Image
import csv
import os
import sys
import shutil
import random
import pandas as pd
import cv2
import decord
from decord import VideoReader, cpu, gpu
decord.bridge.set_bridge('torch')
import numpy as np
import timeit
import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from configuration import build_config
from time import time
import multiprocessing as mp
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
from pku_read_skeleton import read_xyz
from model import build_model
import pdb 


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
    def __init__(self, cfg, input_type, data_split, data_percentage, num_frames, height=270, width=480, skip=0, shuffle=True, transform=None, flag=False, multi_action=False):
        self.cache = {}
        self.dataset = cfg.dataset
        self.flag = flag
        self.multi_action = multi_action
        if self.dataset != "charades":
            self.num_subjects = cfg.num_subjects
        self.data_split = data_split
        self.num_frames = num_frames
        self.videos_folder = cfg.videos_folder
        if data_split == "train":
           self.annotations = cfg.train_annotations
        else:
           self.annotations = cfg.test_annotations
        df = pd.read_csv(self.annotations)
        self.videos = []
        self.subjects = []
        self.data = {}
        self.actions = []
        self.triplets = []
        self.subject_to_videos = {}
        self.views = []
        self.video_actions = {}
        if hasattr(cfg, "skeleton_folder"):
            self.skeleton_files = os.listdir(cfg.skeleton_folder)
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
                    if subject not in self.subjects:
                        self.subjects.append(subject)
                    if action not in self.actions:
                        self.actions.append(action)
                    if view not in self.views:
                        self.views.append(view)
                    if f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}_{view}" not in self.data:
                        self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}_{view}"] = []
                    self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}_{view}"].append([subject, action, video_id, placeholder1, placeholder2, placeholder3, view])
                    
            elif self.dataset == 'mergedntupk':
                if f'{video_id}.hdf5' in hdf5_list or f'{video_id}_{subject}_{action}_{placeholder1}_{placeholder2}.hdf5' in hdf5_list2:
                    if df['subject'].value_counts()[int(subject)] < 2:
                        print('not enough samples: {row}', flush=True)
                        continue
                    self.videos.append([video_id, subject, action, placeholder1, placeholder2, placeholder3])
                    if subject not in self.subjects:
                        self.subjects.append(subject)
                    if action not in self.actions:
                        self.actions.append(action)
                    if f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}" not in self.data:
                        self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"] = []
                    self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"].append([subject, action, video_id, placeholder1, placeholder2, placeholder3])
                    
            elif self.dataset == 'pkummd':
                if int(placeholder1) > int(placeholder2):
                    placeholder1, placeholder2 = placeholder2, placeholder1
                # view_dic = {'L': '001', 'M': '002', 'R': '003'}
                # file_id_ = video_id.split('-')[0][1:]
                # file_view = video_id.split('-')[1]
                # view_id = view_dic[file_view]
                # class_id = int(action)
                # class_id_ = ''
                # if class_id / 10 < 1:
                #     class_id_ = '00' + str(class_id)
                # elif class_id / 100 < 1:
                #     class_id_ = '0' + str(class_id)
                # else:
                #     class_id_ = str(class_id)
                #
                # save_name = 'F' + file_id_ + 'V' + view_id + 'C' + class_id_
                # sk_files = [f for f in self.skeleton_files if save_name in f]
                # # assert len(sk_files) == 1
                # sk_file = sk_files[0]

                if not (('0089' in video_id and action == '23') or ('0043' in video_id and action == '1')):
                    if df['id'].value_counts()[int(subject)] < 2:
                        print('not enough samples: {row}', flush=True)
                        continue
                    self.videos.append([video_id, subject, action, placeholder1, placeholder2, placeholder3])
                    if subject not in self.subjects:
                        self.subjects.append(subject)
                    if action not in self.actions:
                        self.actions.append(action)
                    if f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}" not in self.data:
                        self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"] = []
                    self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"].append([subject, action, video_id, placeholder1, placeholder2, placeholder3])
                else:
                    print(row)

            elif self.dataset == 'numa':
                if f'{video_id[:-4]};{action}.hdf5' in hdf5_list:
                    if df['subject'].value_counts()[int(subject)] < 2:
                        print(row, flush=True)
                        continue
                    self.videos.append([video_id, subject, action, viewpoint])
                    if subject not in self.subjects:
                        self.subjects.append(subject)
                    if action not in self.actions:
                        self.actions.append(action)
                    if f"{subject}_{action}_{video_id}_{viewpoint}" not in self.data:
                        self.data[f"{subject}_{action}_{video_id}_{viewpoint}"] = []
                    self.data[f"{subject}_{action}_{video_id}_{viewpoint}"].append([subject, action, video_id, viewpoint])
                else:
                    print('video not found: {video_id}')

        if shuffle and data_split == 'train':
            random.shuffle(self.videos)
            
        if data_percentage != 1.0:
            len_data = int(len(self.videos) * data_percentage)
            print(len(self.videos), len_data, flush=True)
            self.videos = self.videos[0:len_data]
            print(len(self.videos), flush=True)
        
        self.actions = sorted(self.actions)
        self.views = sorted(self.views)
        print(len(self.subjects), len(self.actions), len(self.videos), len(self.views), self.views, flush=True)
        self.height = height
        self.width = width
        self.skip = skip
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        if self.dataset == 'ntu_rgbd_120' or self.dataset == 'ntu_rgbd_60':
            if self.flag:
                anchor = self.videos[index]
                video_id, sub, act, camera, rep, setup = anchor[0], anchor[1], anchor[2], anchor[3], anchor[4], anchor[5]
                row = [sub, act, video_id, camera, rep, setup]
                
                #view = ';'.join([camera, setup])
                #sv = random.choice([sameview for sameview in self.data.keys() if sameview.split("_")[-1] == view and sameview.split("_")[1] != act])
                #sa = random.choice([sameaction for sameaction in self.data.keys() if sameaction.split("_")[-1] != view and sameaction.split("_")[1] == act])
                
                sv = random.choice([sameview for sameview in self.data.keys() if sameview.split("_")[4] == camera and sameview.split("_")[1] != act])
                sa = random.choice([sameaction for sameaction in self.data.keys() if sameaction.split("_")[4] != camera and sameaction.split("_")[1] == act])
                
                anchor_frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[sa])
                sa_sub, sa_act, sa_video_id, sa_start_frame, sa_end_frame, sa_scene = row[0], row[1], row[2], row[3], row[4], row[5]
                sa_frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[sv])
                sv_sub, sv_act, sv_video_id, sv_start_frame, sv_end_frame =  row[0], row[1], row[2], row[3], row[4]
                sv_frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
              
                action = self.actions.index(act)
                camera = int(camera) - 1
                #camera = self.views.index(view)
                return anchor_frames, sv_frames, sa_frames, camera, action, '_'.join([video_id[8:12], video_id[0:4], video_id[4:8], video_id[12:16], video_id[16:20]])
              
            else:                
                row = self.videos[index]
                video_id, subject, action = row[0], row[1], row[2]
                camera, rep, setup = row[3], row[4], row[5]
                row = [subject, action, video_id, camera, rep, setup]
                view = ';'.join([camera, setup])
                frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                action = self.actions.index(action)
                #camera = self.views.index(view)
                camera = int(camera) - 1   
                return frames, camera, action, '_'.join([video_id[8:12], video_id[0:4], video_id[4:8], video_id[12:16], video_id[16:20]])
            
            
        elif self.dataset == 'pkummd':
            camera_index = ['M', 'R', 'L']
            if self.flag:
                anchor = self.videos[index]
                video_id, sub, act, start_frame, end_frame = anchor[0], anchor[1], anchor[2], anchor[3], anchor[4]
                camera = video_id[-1]
                row = [sub, act, video_id, start_frame, end_frame]
                
                sv = random.choice([sameview for sameview in self.data.keys() if sameview.split("_")[2][-1] == camera and sameview.split("_")[1] != act])
                sa = random.choice([sameaction for sameaction in self.data.keys() if sameaction.split("_")[2][-1] != camera and sameaction.split("_")[1] == act])       
                
                anchor_frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[sa])
                sa_sub, sa_act, sa_video_id, sa_start_frame, sa_end_frame, sa_scene = row[0], row[1], row[2], row[3], row[4], row[5]
                sa_frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[sv])
                sv_sub, sv_act, sv_video_id, sv_start_frame, sv_end_frame =  row[0], row[1], row[2], row[3], row[4]
                sv_frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
              
                camera = camera_index.index(camera)
                action = self.actions.index(act)
                return anchor_frames, sv_frames, sa_frames, camera, action, '_'.join([str(sub), video_id, str(action), str(start_frame), str(end_frame), video_id[-1]])
                
            else:
                row = self.videos[index]
                video_id, subject, action, start_frame, end_frame = row[0], row[1], row[2], int(row[3]), int(row[4])
                camera = video_id[-1]
                row = [subject, action, video_id, start_frame, end_frame]
                frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                action = self.actions.index(action)       
                camera = camera_index.index(camera)
                return frames, camera, action, '_'.join([str(subject), video_id, str(action), str(start_frame), str(end_frame), video_id[-1]])
                
            
        elif self.dataset == 'mergedntupk':
            if self.flag:
                anchor = self.videos[index]
                #print(f'anchor: {anchor}', flush=True)
                video_id, sub, act, p1, p2, p3 = anchor[0], anchor[1], anchor[2], anchor[3], anchor[4], anchor[5]
                #print(f'anchor breakdown: {video_id, sub, act, p1, p2, p3}', flush=True)
                row = [sub, act, video_id, p1, p2, p3]
                sa = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] == act and diff_sub.split("_")[0] != sub])
                #print(f'sa: {sa}', flush=True)
                ss = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] != act and diff_sub.split("_")[0] == sub])
                #print(f'ss: {ss}', flush=True)
                
                anchor_frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform, self.blurred_model, self.bcfg)
                
                row = random.choice(self.data[sa])
                #print(f'row: {row}', flush=True)
                sa_sub, sa_act, sa_video_id, sa_start_frame, sa_end_frame, sa_scene = row[0], row[1], row[2], row[3], row[4], row[5]
                sa_frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform, self.blurred_model, self.bcfg)
                
                row2 = random.choice(self.data[ss])
                #print(f'row2: {row2}', flush=True)
                ss_sub, ss_act, ss_video_id, ss_start_frame, ss_end_frame =  row2[0], row2[1], row2[2], row2[3], row2[4]
                ss_frames = self.frame_creation(row2, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform, self.blurred_model, self.bcfg)
                
                #print(f'anchor: {anchor}, sa: {row}, ss: {row2}', flush=True)
              
                subject = self.subjects.index(sub)
                action = self.actions.index(act)
                
                
                if video_id[0] == 'S':
                    return anchor_frames, ss_frames, sa_frames, subject, action, '_'.join([video_id[8:12], video_id[0:4], video_id[4:8], video_id[12:16], video_id[16:20]])
                else:
                    return anchor_frames, ss_frames, sa_frames, subject, action, '_'.join([str(subject), video_id, str(action), str(p1), str(p2), video_id[-1]])
              
            else:
                row = self.videos[index]
                video_id, subject, action = row[0], row[1], row[2]
                p1, p2, p3 = row[3], row[4], row[5]
                row = [subject, action, video_id, p1, p2, p3]
                frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform, self.blurred_model, self.bcfg)
                action = self.actions.index(action)       
                subject = self.subjects.index(subject)
                if video_id[0] == 'S':
                    return frames, subject, action, '_'.join([video_id[8:12], video_id[0:4], video_id[4:8], video_id[12:16], video_id[16:20]])
                else:
                    return frames, subject, action, '_'.join([str(subject), video_id, str(action), str(p1), str(p2), video_id[-1]])
                    
                    
        elif self.dataset == 'numa':
            if self.flag:
                anchor = self.videos[index]
                video_id, sub, act, viewpoint = anchor[0], anchor[1], anchor[2], anchor[3]
                row = [sub, act, video_id, viewpoint]
                print(row)
                sv = random.choice([sameview for sameview in self.data.keys() if sameview.split("_")[-1] == viewpoint and sameview.split("_")[1] != act])
                sa = random.choice([sameaction for sameaction in self.data.keys() if sameaction.split("_")[-1] != viewpoint and sameaction.split("_")[1] == act])
                
                anchor_frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                row = random.choice(self.data[sa])
                print(row)
                sa_sub, sa_act, sa_video_id, sa_viewpoint = row[0], row[1], row[2], row[3]
                sa_frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[sv])
                print(row)
                sv_sub, sv_act, sv_video_id, sv_viewpoint =  row[0], row[1], row[2], row[3]
                sv_frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
              
                action = self.actions.index(act)
                print(act, action, self.actions)
                viewpoint = int(viewpoint) - 1 
                return anchor_frames, sv_frames, sa_frames, viewpoint, action, '_'.join([str(sub), video_id, str(action), str(viewpoint), video_id[9:11]])
              
            else:
                row = self.videos[index]
                video_id, subject, action, viewpoint = row[0], row[1], row[2], row[3]
                row = [subject, action, video_id, viewpoint]
                frames = self.frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                action = self.actions.index(action)
                viewpoint = int(viewpoint) - 1     
                return frames, viewpoint, action, '_'.join([str(subject), video_id, str(action), str(viewpoint), video_id[9:11]])
            
    def frame_creation(self, row, dataset, videos_folder, height, width, num_frames, transform, blurred_model=None, bcfg=None):    
        if dataset == "ntu_rgbd_120" or dataset == 'ntu_rgbd_60':
            list16 = []
            subject, action, video_id, start_frame, end_frame = row[0], row[1], row[2], row[3], row[4]
            frames = h5py.File(os.path.join('/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120', f'{video_id}.hdf5'), 'r')
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
                    frames = h5py.File(os.path.join('/home/siddiqui/Action_Biometrics/frame_data/pkummd', f'{video_id}_{int(subject)-1}_{action}_{start_frame}_{end_frame}.hdf5'), 'r')
                except OSError:
                    frames = h5py.File(os.path.join('/home/siddiqui/Action_Biometrics/frame_data/pkummd', f'{video_id}_{int(subject)+8}_{action}_{start_frame}_{end_frame}.hdf5'), 'r')
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
                path = '/home/c3-0/praveen/datasets/PKUMMD/data/skeleton_ntu'
                view_dic = {'L':'001','M':'002','R':'003'}
                file_id_ = video_id.split('-')[0][1:]
                file_view = video_id.split('-')[1]
                view_id = view_dic[file_view]
                
                class_id = int(action)
                class_id_ = ''
                if class_id / 10 < 1:
                    class_id_ = '00' + str(class_id)
                elif class_id / 100 < 1:
                    class_id_ = '0' + str(class_id)
                else:
                    class_id_ = str(class_id)

                save_name = 'F' + file_id_ + 'V' + view_id + 'C' + class_id_
                sk_files = [f for f in self.skeleton_files if save_name in f]
                # assert len(sk_files) == 1
                sk_file = sk_files[0]
                
                skeleton_file = os.path.join(path, sk_file)
                skeletons = read_xyz(skeleton_file)
                
                frame_indexer = np.linspace(0, skeletons.shape[1] - 1, 16).astype(int)
                action_skeletons = skeletons[:, frame_indexer, :, :]
                # print(action_skeletons.shape)
                return action_skeletons
            
        elif dataset == 'mergedntupk':
            blurred = False
            list32 = []
            subject, action, video_id, start_frame, end_frame = row[0], row[1], row[2], int(row[3]), int(row[4])
            start = timeit.default_timer()
            if video_id[0] == 'S':
                frames = h5py.File(os.path.join('/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120', f'{video_id}.hdf5'), 'r')
            else:
                frames = h5py.File(os.path.join('/home/siddiqui/Action_Biometrics/frame_data/pkummd', f'{video_id}_{subject}_{action}_{start_frame}_{end_frame}.hdf5'), 'r')
            frames = frames['default'][()]
            frames = torch.from_numpy(frames).float()
            
            if video_id[0] != 'S':
                frame_indexer = np.linspace(start_frame, end_frame - 1, 16).astype(int)
                for i, frame in enumerate(frames, start_frame):
                    if i in frame_indexer:
                        list32.append(frame)
                frames = torch.stack([frame for frame in list32])
            else:
                frame_indexer = np.linspace(0, 31, 16).astype(int)
                for i, frame in enumerate(frames):
                    if i in frame_indexer:
                        list32.append(frame)
                frames = torch.stack([frame for frame in list32])
                
                
            for i, frame in enumerate(frames):
                frames[i] = frames[i] / 255.
                
            if transform:
                frames = frames.transpose(0, 1)
                frames = transform(frames)
                frames = frames.transpose(0, 1)
            #print(f'transform + final time {timeit.default_timer() - start}', flush=True)
            #torch.save(frames, 'blurred_final.pt')
            return frames
            
        elif dataset == "numa":
            list16 = []
            subject, action, video_id, viewpoint = row
            with h5py.File(f'/home/siddiqui/Action_Biometrics/frame_data/numa/{video_id[:-4]};{action}.hdf5', 'r') as f:
                frames = f['default'][:]
                frames = torch.from_numpy(frames)
                frames = frames.type(torch.float32)
                
            frame_indexer = np.linspace(0, len(frames)-1, num_frames).astype(int)
            
            #print(len(frames), frame_indexer, flush=True)
            
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
    #frames = frames = h5py.File('/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120/S001C001P001R001A010_rgb.avi.hdf5')
    #frames = frames['default'][()]
    #print(frames.shape)
    #net, bcfg = load_model()
    #blur = torch_blur(frames, net, bcfg)
    #torch.save(blur, 'blurdr.pt')

    cs_skeleton_action_list = [73, 74, 72, 71, 12, 29, 106, 75, 76, 105]
    cv_skeleton_action_list = [73, 12, 76, 72, 74, 11, 103, 75, 71, 105]

    data_generator = omniDataLoader(cfg, 'rgb', 'train', 1.0, 16, skip=0, shuffle=shuffle, transform=None, flag=False, multi_action=False)
    dataloader = DataLoader(data_generator, batch_size=1, num_workers=0, shuffle=False, drop_last=True, collate_fn=None)
    
    for (clips, views, actions, keys) in tqdm(dataloader):
        print(clips.shape, actions.shape, views.shape)
        exit()
    
   
    #for (clips, sv_clips, sa_clips, views, actions, keys) in tqdm(dataloader):
    #    print(clips.shape, sv_clips.shape, sa_clips.shape, views, actions)
    #    print(actions.shape, views.shape)
    #    exit()
    

        