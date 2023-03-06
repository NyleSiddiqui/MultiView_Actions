import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_uniform_
import torch
import torchvision.models.video as video_models
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
from models.v1_backbone import VideoTransformer as V1
from models.v3_backbone import VideoTransformer as V3
from models.v1_skeleton import VideoTransformer as Skeleton
from models.v1_skeleton2 import VideoTransformer as Skeleton2



def cosine_pairwise_dist(x, y):
    assert x.shape[1] == y.shape[1], "both sets of features must have same shape"
    return nn.functional.cosine_similarity(x[:, None, :], y[None, :, :], dim=-1)  

def weights_init(m):
    if isinstance(m, nn.Linear):
        kaiming_uniform_(m.weight.data)

def build_model(version, input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers):
    if version == 'v3':
        model = V3(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    elif version == 'v1':
        model = V1(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    elif version == 'skeleton':
        model = Skeleton(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    elif version == 'skeleton2':
        model = Skeleton2(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    model.apply(weights_init)
    return model


#i3d params: 13,541,352
#r3d params: 33,323,250
#v3 2 layers: 15,773,386

if __name__ == '__main__':
    layers = 2
    model = build_model('skeleton2', 150, 16, 3, 41, 16, 256, 8, layers)
    model.cuda()
    
    total_params = sum(p.numel() for p in model.parameters()) 
    print(total_params)

    model.eval()
    #features = Variable(torch.rand(2, 16, 150)).cuda()

    N, C, T, V, M = 6, 3, 16, 25, 2
    features = Variable(torch.randn(N, C, T, V, M)).cuda()
    
    output_subject, output_action, m_features, act_features, actq, subq = model(features)
    print(output_subject.shape, output_action.shape, m_features.shape, act_features.shape, flush=True)
    exit()
    




