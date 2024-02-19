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
from models.v3_backbone import VideoTransformer as V3


def cosine_pairwise_dist(x, y):
    assert x.shape[1] == y.shape[1], "both sets of features must have same shape"
    return nn.functional.cosine_similarity(x[:, None, :], y[None, :, :], dim=-1)  

def weights_init(m):
    if isinstance(m, nn.Linear):
        kaiming_uniform_(m.weight.data)

def build_model(version, num_views, num_actions):
    if version == 'v3':
        model = V3(num_views, num_actions)
    model.apply(weights_init)
    return model


#i3d params: 13,541,352
#r3d params: 33,323,250
#v3 2 layers: 15,773,386

if __name__ == '__main__':
    layers = 2
    model = build_model('v1', 224, 3, 41)
    model.cuda()
    
    total_params = sum(p.numel() for p in model.parameters()) 
    print(total_params)

    model.eval()
    features = Variable(torch.rand(2, 16, 3, 224, 224)).cuda()
    
    output_subject, output_action, m_features, act_features, actq, subq = model(features)
    print(output_subject.shape, output_action.shape, m_features.shape, act_features.shape, flush=True)
    




