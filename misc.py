import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
import h5py
import cv2
#from torch import nn, einsum
#from einops import rearrange, repeat
import timeit
from decord import VideoReader, cpu
import torch.nn.functional as F
#from imageio import imsave
from torchvision.utils import save_image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
    Resize,
    PILToTensor
    
)

def splitActions(df):
    df.drop(['quality', 'relevance', 'script', 'objects', 'descriptions', 'verified'], axis=1, inplace=True)
    df = df.dropna(subset = ['actions'])
    new_df_rows = []
    for count, row in enumerate(df.to_numpy()):
        try:
            ast = row[3].split(';')
        except AttributeError:
            ast = row[3]
        for actions in ast:
            action, start, end = actions.split(' ')
            new_df_rows.append([row[0], row[1], row[2], action[1:], start, end])
    new_df = pd.DataFrame(new_df_rows, columns=['id', 'subject', 'scene', 'action', 'start', 'end'])
    new_df.reset_index()
    new_df.to_csv("CharadesEgo_test.csv")


def indexNTU():
    path = "/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb"
    train_df_rows = []
    test_df_rows = []
    for row in os.listdir(path):
        if row[16:20] in ['A050', 'A051', 'A052', 'A053', 'A054', 'A055', 'A056', 'A057', 'A058',
                          'A059', 'A060', 'A106', 'A107', 'A108', 'A109', 'A110', 'A111', 'A112',
                          'A113', 'A114', 'A115', 'A116', 'A117', 'A118', 'A119', 'A120']:
            continue
        else:
            s_num, cam_id, sub_id, rep_num, act_id = row[0:4], row[4:8], row[8:12], row[12:16], row[16:20]
            if int(sub_id[1:]) in range(1, 71):
                train_df_rows.append([row, s_num[1:], cam_id[1:], sub_id[1:], rep_num[1:], act_id[1:]])
            else:
                test_df_rows.append([row, s_num[1:], cam_id[1:], sub_id[1:], rep_num[1:], act_id[1:]])
                
    df = pd.DataFrame(train_df_rows, columns=['video_id', 'setup', 'camera', 'subject', 'repetition', 'action'])
    df.reset_index()
    df.to_csv("/home/siddiqui/Action_Biometrics/data/NTUTrain_map2.csv")
    
    df = pd.DataFrame(test_df_rows, columns=['video_id', 'setup', 'camera', 'subject', 'repetition', 'action'])
    df.reset_index()
    df.to_csv("/home/siddiqui/Action_Biometrics/data/NTUTest_map2.csv")
    
    
def indexNTUupdated():
    path = "/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb"
    train_df_rows = []
    test_df_rows = []
    train_subs = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
                  80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
    count = 0
    print(len(train_subs))
        
    for video in os.listdir(path):
        #print(int(video[1:4]))
        #print(int(video[9:12]))
        #print(video, int(video[9:12]), int(video[17:20]), int(video[5:8]), int(video[13:16]), int(video[1:4]))
        #exit()
        if int(video[1:4]) % 2 == 0:
            train_df_rows.append([video, int(video[9:12]), int(video[17:20]), int(video[5:8]), int(video[13:16]), int(video[1:4])])
        else:
            test_df_rows.append([video, int(video[9:12]), int(video[17:20]), int(video[5:8]), int(video[13:16]), int(video[1:4])])

                
    df = pd.DataFrame(train_df_rows, columns=['video_id', 'subject', 'action', 'camera', 'repetition', 'setup'])
    df.to_csv("/home/siddiqui/Action_Biometrics/data/NTUTrain_CVmap.csv", index=False)
    
    df = pd.DataFrame(test_df_rows, columns=['video_id', 'subject', 'action', 'camera', 'repetition', 'setup'])
    df.to_csv("/home/siddiqui/Action_Biometrics/data/NTUTest_CVmap.csv", index=False)
    
    
    

def indexPK():
    path = "C://Users/ny525072/Downloads/label"
    trdf_rows = []
    ttdf_rows = []
    for count, text_file in enumerate(os.listdir(path)):
        sub_id = int(text_file[4:6])   # + 66
        for line in open(os.path.join(path, text_file), 'r').readlines():
            if len(line.split(',')) == 4:
                act_id, start_frame, end_frame, confidence = line.split(',')
                confidence = confidence[0]
            else:
                act_id, start_frame, end_frame = line.split(',')
                confidence = 2
            if sub_id in range(1, 11):
                trdf_rows.append([f"{text_file[:-4]}", sub_id, act_id, start_frame, end_frame, confidence])
            else:
                ttdf_rows.append([f"{text_file[:-4]}", sub_id, act_id, start_frame, end_frame, confidence])
    df = pd.DataFrame(trdf_rows, columns=['video_id', 'id', 'action', 'start', 'end', 'confidence'])
    df.reset_index()
    df.to_csv("PKUMMDv2Train_map.csv")
    df2 = pd.DataFrame(ttdf_rows, columns=['video_id', 'id', 'action', 'start', 'end', 'confidence'])
    df2.reset_index()
    df2.to_csv("PKUMMDv2Test_map.csv")


def mergePK():
    # path = "/home/siddiqui/Action_Biometrics/data"
    v1train = pd.read_csv("PKUMMDv1Train_map.csv")
    v1test = pd.read_csv("PKUMMDv1Test_map.csv")

    v2train = pd.read_csv("PKUMMDv2Train_map.csv")
    v2test = pd.read_csv("PKUMMDv2Test_map.csv")
    v2train["id"] += 66
    v2test["id"] += 66

    train = pd.concat([v1train, v2train])
    test = pd.concat([v1test, v2test])
    train.to_csv("MPKUMMDTrain_map.csv")
    test.to_csv("MPKUMMDTest_map.csv")

def processPKUMMD():
    new_df_rows = []
    targets = pd.read_excel("C://Users/nyles/Downloads/targets.xlsx")
    sublabels = dict(zip(targets["Index"], targets["Subject ID"]))
    path = "C://Users/ny525072/Downloads"
    #path = "/home/c3-0/datasets/PKUMMD/LABELS/Train_Label_PKU_final/"
    for text in os.listdir(path):
        filedata = pd.read_csv(f"{os.path.join(path, text)}", header=None)
        for row in filedata.to_numpy():
            action, start, end, confidence = row[0], row[1], row[2], row[3]
            sub_id = sublabels[int(text[:4])]
            new_df_rows.append([text[:-4], action, start, end, confidence, sub_id])
    new_df = pd.DataFrame(new_df_rows, columns=["video", "action", "start", "end", "confidence", "id"])
    new_df.reset_index()
    new_df.to_csv("PKUMMD_map.csv")

def renamefiles():
    path = "home/c3-0/datasets/MergedPKUMMD/RGB_VIDEO"
    for file in os.listdir(path):
        if len(file) > 11:
            os.rename(os.path.join(path, file), os.path.join(path, f"{file[:-10]}"))

def fixsubjectid():
    df = pd.read_csv("MPKUMMDTrain_map2.csv")
    df2 = pd.read_csv("MPKUMMDTest_map2.csv")
    new_df = df.values
    new_df2 = df2.values
    for row in new_df:
        row[1] -= 1
    for row in new_df2:
        row[1] -= 1
    df = pd.DataFrame(new_df, columns=['video_id', 'id', 'action', 'start', 'end', 'confidence'])
    df2 = pd.DataFrame(new_df2, columns=['video_id', 'id', 'action', 'start', 'end', 'confidence'])
    print(np.unique(df["id"]), np.unique(df2["id"]))
    df.to_csv("MPKUMMDTrain_map3.csv")
    df2.to_csv("MPKUMMDTest_map3.csv")


def fixPKindex():
    videos = []
    subjects = []
    actions = []
    data = {}
    for row in open("NTUTrain_map.csv", 'r').readlines()[1:]:
        if len(row.split(',')) == 6:
            video_id, subject, action, placeholder1, placeholder2, placeholder3 = row.split(',')
            videos.append([video_id, subject, action, placeholder1, placeholder2, placeholder3])
            if subject not in subjects:
                subjects.append(subject)
            if action not in actions:
                actions.append(action)
            if f"{subject}_{action}" not in data:
                data[f"{subject}_{action}"] = []
            data[f"{subject}_{action}"].append([video_id, subject, action, placeholder1, placeholder2, placeholder3])
    for key in data.keys():
        subject, action = key.split("_")[0], key.split("_")[1]
        if int(action) == 18 or int(action) == 21:
            continue
        new_action = random.choice([diff_action for diff_action in data.keys() if diff_action.split("_")[1] != action and diff_action.split("_")[0] == subject])
        new_subject = random.choice([diff_sub for diff_sub in data.keys() if diff_sub.split("_")[1] == action and diff_sub.split("_")[0] != subject])
        same_subject_vid = random.choice(data[new_action])
        same_action_vid = random.choice(data[new_subject])
        print(video_id[0:4], video_id[:4])
        ss_video_id, ss_action = same_subject_vid[0], same_subject_vid[2]
        sa_video_id, sa_subject = same_action_vid[0], same_action_vid[1]


def small_NTU():
    path = "/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_+rgb"
    train = []
    test = []
    for row in os.listdir(path):
        if int(row[8:12]) not in range(1, 28) or row[16:20] in ['A050', 'A051', 'A052', 'A053', 'A054', 'A055', 'A056', 'A057', 'A058',
                          'A059', 'A060', 'A106', 'A107', 'A108', 'A109', 'A110', 'A111', 'A112',
                          'A113', 'A114', 'A115', 'A116', 'A117', 'A118', 'A119', 'A120']:
            continue
        else:
            s_num, cam_id, sub_id, rep_num, act_id = row[0:4], row[4:8], row[8:12], row[12:16], row[16:20]
            if sub_id in range(1, 20):
                train.append([row, s_num[1:], cam_id[1:], sub_id[1:], rep_num[1:], act_id[1:]])
            else:
                test.append([row, s_num[1:], cam_id[1:], sub_id[1:], rep_num[1:], act_id[1:]])
    df = pd.DataFrame(train, columns=['video_id', 'setup', 'camera', 'subject', 'repetition', 'action'])
    df.reset_index()
    df.to_csv("/home/siddiqui/Action_Biometrics/data/SmallNTUTrain_map.csv")

    df = pd.DataFrame(test, columns=['video_id', 'setup', 'camera', 'subject', 'repetition', 'action'])
    df.reset_index()
    df.to_csv("/home/siddiqui/Action_Biometrics/data/SmallNTUTest_map.csv")
    
    
def video_to_hp5y(setting):
    if setting == "ntu":
        anno = pd.read_csv("data/NTUTrain_map.csv")
        path = "/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb"
        resize = Resize([270, 480])
        frames = []
        for i, video in enumerate(anno['video_id']):
            start = timeit.default_timer()
            if i % 100 == 0:
                print(i, flush=True)
            if os.path.exists(f'/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120/{video}.hdf5'):
                print(f'/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120/{video}.hdf5 already exists!, {i}', flush=True)
                continue
            count = 0
            if ".avi" in video:
                start = timeit.default_timer()
                cap = cv2.VideoCapture(os.path.join(path, video))
                length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                action_length = length / 24
                fps10 = round(action_length * 10)
                if fps10 > 32:
                    frame_ids = np.linspace(0, length - 1, fps10).astype(int)
                    frame_ids = np.linspace(0, length - 1, 32).astype(int)
                else:
                    frame_ids = np.linspace(0, length - 1, 32).astype(int)
                ret, frame = cap.read()
                while ret:
                    if count in frame_ids:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = torch.as_tensor(frame)
                        frame = frame.permute(2, 0, 1)
                        frame = resize(frame)
                        frames.append(frame)
                    count += 1
                    ret, frame = cap.read()
                tframes = torch.stack([frame for frame in frames])
                frames.clear()
            print(f"one video time: {timeit.default_timer() - start}", flush=True)
            if not os.path.exists(f'/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120/{video}.hdf5'):
                with h5py.File(f'/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120/{video}.hdf5', 'w') as f:
                        dset = f.create_dataset('default', data=tframes)
            else:
                print(f'/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120/{video}.hdf5 already exists!, {i}', flush=True)
            del tframes
            
            
    elif setting == 'pk':
        anno = pd.read_csv("/home/siddiqui/Action_Biometrics/data/MPKUMMDTrain_map4edit.csv")
        path = "/home/c3-0/datasets/MergedPKUMMD/RGB_VIDEO/"
        resize = Resize([270, 480])
        frames = []
        for i, row in enumerate(anno.values):
            video_id = row[0]
            subject = row[1]
            action = row[2]
            start, end = int(row[3]), int(row[4])
            if end < start:
                start, end = end, start
            count = 0
            if os.path.exists(f'/home/siddiqui/Action_Biometrics/frame_data/pkummd/{video_id}_{subject}_{action}_{start}_{end}.hdf5'):
                print(i)
                continue
            if os.path.exists(f'/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120/{video_id}.hdf5'):
                print(i)
                continue
            start_timer = timeit.default_timer()
            cap = cv2.VideoCapture(os.path.join(path, f'{video_id}.avi'))
            cap.set(1, start)
            #frame_indexer = np.linspace(start, end - 1, 32).astype(int)
            #print(frame_indexer, flush=True)
            ret, frame = cap.read()
            while ret and count+start < end:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frame = resize(frame)
                frames.append(frame)
                count += 1
                ret, frame = cap.read()
            tframes = torch.stack([frame for frame in frames])
            print(tframes.shape, flush=True)
            frames.clear()
            print(f"one video time: {timeit.default_timer() - start_timer}", flush=True)
            if not os.path.exists(f'/home/siddiqui/Action_Biometrics/frame_data/pkummd/{video_id}_{subject}_{action}_{start}_{end}.hdf5'):
                with h5py.File(f'/home/siddiqui/Action_Biometrics/frame_data/pkummd/{video_id}_{subject}_{action}_{start}_{end}.hdf5', 'w') as f:
                        dset = f.create_dataset('default', data=tframes)
            else:
                print(f'/home/siddiqui/Action_Biometrics/frame_data/pkummd/{video_id}_{subject}_{action}_{start}_{end}.hdf5 already exists!', flush=True)
            if i % 50 == 0:
                print(i, flush=True)
            del tframes
    
    elif setting == "numa":
        anno = pd.read_csv("NUMAMaster.csv")
        path = "/home/c3-0/datasets/NUMA/multiview_action_videos"
        resize = Resize([270, 480])
        frames = []
        for i, row in enumerate(anno.values):
            video, subject, action, viewpoint = row
            if os.path.exists(f'/home/siddiqui/Action_Biometrics/frame_data/numa/{video[:-4]};{action}.hdf5'):
                print(f'/home/siddiqui/Action_Biometrics/frame_data/numa/{video[:-4]};{action}.hdf5 already exists!, {i}', flush=True)
                continue
            start = timeit.default_timer()
            if i % 100 == 0:
                print(i, flush=True)
            count = 0
            if ".avi" in video:
                start = timeit.default_timer()
                cap = cv2.VideoCapture(os.path.join(path, f'a{str(action).zfill(2)}', video))
                length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                ret, frame = cap.read()
                while ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.as_tensor(frame)
                    frame = frame.permute(2, 0, 1)
                    frame = resize(frame)
                    frames.append(frame)
                    ret, frame = cap.read()
                tframes = torch.stack([frame for frame in frames])
                frames.clear()
            print(f"one video time: {timeit.default_timer() - start}, {video}", flush=True)
            if not os.path.exists(f'/home/siddiqui/Action_Biometrics/frame_data/numa/{video[:-4]};{action}.hdf5'):
                with h5py.File(f'/home/siddiqui/Action_Biometrics/frame_data/numa/{video[:-4]};{action}.hdf5', 'w') as f:
                        dset = f.create_dataset('default', data=tframes)
            del tframes 
            
            
def h5py_to_tensor():
    file_folder = '/home/siddiqui/Action_Biometrics/frame_data/PK/'
    for file in os.listdir(file_folder):
        if '.hdf5' in file:
            loadfile = timeit.default_timer()
            frames = h5py.File(os.path.join(file_folder, file), 'r')
            print(frames['default'].shape, flush=True)
            print(type(frames['default']), flush=True)
            print(f"loadfile: {timeit.default_timer() - loadfile}", flush=True)
            start = timeit.default_timer()
            frames = torch.tensor(frames['default'])
            print(f"tensor: {timeit.default_timer() - start}", flush=True)
            print(frames.shape, flush=True)
            

def save():
    image_folder = '/squash/Charades_Charades_v1_rgb/001YG'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 24, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    video.release()
    
    
    
def mergeNTUPK():
    merged_df_rows = []
    same_actions_ntu_to_pk = {35:1, 4:2, 3:3, 33:4, 22:5, 10:6, 40:7, 1:8, 5:9, 2:10, 23:13, 26:15, 27:17, 24:19, 28:20, 6:22, 29:23, 31:25, 20:28, 25:29, 11:30, 34:31, 38:32, 8:33, 9:34, 21:35, 19:36, 15:37, 32:39, 13:40, 7:41, 46:42, 45:43, 44:44, 47:45, 30:46, 49:47, 14:48, 18:49, 37:50, 12:51}
    print(same_actions_ntu_to_pk.values())
    
    trainpk = pd.read_csv("/home/siddiqui/Action_Biometrics/data/MPKUMMDTrain_map4edit.csv")
    testpk = pd.read_csv("/home/siddiqui/Action_Biometrics/data/MPKUMMDTest_map4edit.csv")
    masterntu = pd.read_csv("/home/siddiqui/Action_Biometrics/data/NTUMaster_map2.csv")
    
#    path = "/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb/"
#
#    for row in os.listdir(path):
#        if int(row[17:20]) in same_actions_ntu_to_pk.keys(): 
#            s_num, cam_id, sub_id, rep_num, act_id = row[0:4], row[4:8], row[8:12], row[12:16], row[16:20]
#            master.append([row, s_num[1:], cam_id[1:], sub_id[1:], rep_num[1:], act_id[1:]])
#                
#    df = pd.DataFrame(master, columns=['video_id', 'setup', 'camera', 'subject', 'repetition', 'action'])
#    df.reset_index()
#    df.to_csv("/home/siddiqui/Action_Biometrics/data/NTUMaster_map2.csv")
    
    for row in trainpk.values:
        if row[2] in same_actions_ntu_to_pk.values():
            merged_df_rows.append(row)
            
    for row in testpk.values:
        if (row[2]) in same_actions_ntu_to_pk.values():
            merged_df_rows.append(row)
    print('pk done')
        
    for row in masterntu.values:
        if row[2] in same_actions_ntu_to_pk.values():
            row[1] += 100
            row[2] = same_actions_ntu_to_pk[row[2]]
            merged_df_rows.append(row)
    merged_df = pd.DataFrame(merged_df_rows, columns=['video_id', 'subject', 'action', 'p1', 'p2', 'p3'])
    merged_df.to_csv("/home/siddiqui/Action_Biometrics/data/MergedNTUPKMaster_map.csv")
    
    
def split_mergedNTUPK():
    train_df = [] 
    test_df = []
    master = pd.read_csv("/home/siddiqui/Action_Biometrics/data/MergedNTUPKMaster_map.csv")
    
    
    for i, row in enumerate(master.values):
        if i in range(19255) or i in range(25174, 48022):
            train_df.append(row)
        else:
            test_df.append(row)
            
    tr_df = pd.DataFrame(train_df, columns=['video_id', 'subject', 'action', 'p1', 'p2', 'p3'])
    tr_df.to_csv("/home/siddiqui/Action_Biometrics/data/MergedNTUPKTrain_map.csv")
    
    tt_df = pd.DataFrame(test_df, columns=['video_id', 'subject', 'action', 'p1', 'p2', 'p3'])
    tt_df.to_csv("/home/siddiqui/Action_Biometrics/data/MergedNTUPKTest_map.csv")
    
            
def indexNUMA():
    df_rows = []
    path = '/home/c3-0/datasets/NUMA/multiview_action_videos/'
    for folder in os.listdir(path):
        for vid in os.listdir(os.path.join(path, folder)):
            if '.avi' in vid:
                viewpoint, subject, _ = vid.split('_')
                
                action = int(folder[1:])
                subject = int(subject[1:])
                viewpoint = int(viewpoint[1:])
                
                #print(vid, subject, action, viewpoint)
                df_rows.append([vid, subject, action, viewpoint])
    df = pd.DataFrame(df_rows, columns = ['video', 'subject', 'action', 'viewpoint'])
    df.to_csv('NUMAMaster.csv', index=False)
    
    
def splitNUMA():
    train_df_rows = []
    test_df_rows = []
    for i, row in enumerate(open('NUMAMaster.csv', 'r').readlines()[1:]):
        video, subject, action, view = row.split(',')
        if int(view) < 3:
            train_df_rows.append([video, subject, action, view])
        else:
            test_df_rows.append([video, subject, action, view])
        
    df = pd.DataFrame(train_df_rows, columns = ['video', 'subject', 'action', 'viewpoint'])
    df.to_csv('NUMATrain_CV.csv', index=False)
    
    df = pd.DataFrame(test_df_rows, columns = ['video', 'subject', 'action', 'viewpoint'])
    df.to_csv('NUMATest_CV.csv', index=False)
        
            
            
            
if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    #video_to_hp5y('ntu')
    video = 'S007C001P017R002A058_rgb.avi'
    path = '/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb'
    resize = Resize([270, 480])
    frames = []
    count = 0
    if ".avi" in video:
        start = timeit.default_timer()
        cap = cv2.VideoCapture(os.path.join(path, video))
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        action_length = length / 24
        fps10 = round(action_length * 10)
        if fps10 > 32:
            frame_ids = np.linspace(0, length - 1, fps10).astype(int)
            frame_ids = np.linspace(0, length - 1, 32).astype(int)
        else:
            frame_ids = np.linspace(0, length - 1, 32).astype(int)
        ret, frame = cap.read()
        while ret:
            if count in frame_ids:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.as_tensor(frame)
                frame = frame.permute(2, 0, 1)
                frame = resize(frame)
                frames.append(frame)
            count += 1
            ret, frame = cap.read()
        tframes = torch.stack([frame for frame in frames])
        frames.clear()
    print(f"one video time: {timeit.default_timer() - start}", flush=True)
    if not os.path.exists(f'/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120/{video}.hdf5'):
        with h5py.File(f'/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120/{video}.hdf5', 'w') as f:
                dset = f.create_dataset('default', data=tframes)
    del tframes





