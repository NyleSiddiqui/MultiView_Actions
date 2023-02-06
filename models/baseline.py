import torch 
from torch import nn
from torchvision.models.video import r2plus1d_18, r3d_18, R2Plus1D_18_Weights, R3D_18_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

class VideoTransformer(nn.Module):
    def __init__(self, input_size, num_frames, num_subjects, patch_size, hidden_dim, num_heads, num_layers):
        super(VideoTransformer, self).__init__()
        self.input_size = input_size
        self.num_frames = num_frames
        self.num_subjects = num_subjects
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.vivit_model = ViViT(self.input_size, self.patch_size, self.num_subjects, self.num_frames, dim=hidden_dim).cuda()
        self.classifier = nn.Linear(self.hidden_dim, self.num_subjects)

    def forward(self, inputs):
        outputs, actions, features = self.vivit_model(inputs)
        return outputs, actions, features


class I3D(nn.Module):
    def __init__(self, num_subjects, num_actions, hidden_dim):
        super(I3D, self).__init__()
        self.num_subjects = num_subjects
        self.num_actions = num_actions
        self.I3D_model = InceptionI3d(num_classes=self.num_subjects, num_actions=self.num_actions, hidden_dim=hidden_dim)
        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        outputs, actions, features = self.I3D_model(inputs)
        return outputs, actions, features


class I3DBackbone(nn.Module):
    def __init__(self, pretrained, num_subjects=100, hidden_dim=1024):
        super(I3DBackbone, self).__init__()
        self.model = InceptionI3dBackbone(num_classes=num_subjects).cuda()
        self.hidden_dim = hidden_dim
        self.avg_pool = nn.AvgPool3d(kernel_size=[1, 7, 7], stride=(1, 1, 1))
        self.name = 'i3d'
        
        if pretrained:
            model_kvpair = self.model.state_dict()
            for layer_name, weights in torch.load('/home/siddiqui/Action_Biometrics/results/saved_models/mergedNTUPK+I3D_12-01-23_1317/model_5_66.4150.pth')['state_dict'].items():
                layer_name = layer_name.replace('I3D_model.','')
                layer_name = layer_name.replace('extractor.','')
                if 'logits' in layer_name or 'actions' in layer_name or 'feature_dim' in layer_name or 'features' in layer_name:
                    continue
                model_kvpair[layer_name]=weights
            self.model.load_state_dict(model_kvpair, strict=True)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        features = self.model.extract_features(inputs)
        features = self.avg_pool(features)
        return features


class R2plus1D(nn.Module):
    def __init__(self, num_subjects):
        super(R2plus1D, self).__init__()
        weights = R2Plus1D_18_Weights.DEFAULT
        self.num_subjects = num_subjects
        model = r2plus1d_18(weights=weights).cuda()
        model.fc = nn.Linear(512, self.num_subjects)
        self.R2plus1D_model = model
        self.extractor = create_feature_extractor(model, return_nodes={"avgpool": "features"})
        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        outputs = self.R2plus1D_model(inputs)
        features = self.extractor(inputs)
        features = list(features.values())
        features = features[0]
        features = torch.squeeze(features)
        return outputs, features
        
        
class R3D(nn.Module):
    def __init__(self, num_subjects, num_actions, hidden_dim):
        super(R3D, self).__init__()
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights).cuda()
        #model = r3d_18().cuda()
        self.num_subjects = num_subjects
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        model.fc = nn.Linear(512, self.num_subjects)
        self.R3D_model = model
        self.extractor = create_feature_extractor(model, return_nodes={"avgpool": "features"})
        self.actions_head = nn.Linear(512, self.num_actions)
        self.subjects_head = nn.Linear(512, self.num_subjects)

        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        features = self.extractor(inputs)
        features = list(features.values())
        features = features[0]
        features = torch.squeeze(features)
        outputs = self.subjects_head(features)
        actions = self.actions_head(features)
        return outputs, actions, features
        
        
class R3DBackbone(nn.Module):
    def __init__(self, pretrained, hidden_dim=512):
        super(R3DBackbone, self).__init__()
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights).cuda()
        self.hidden_dim = hidden_dim
        self.R3D_model = model
        self.extractor = create_feature_extractor(model, return_nodes={"layer4": "features"}) #extract from earlier layer?
        self.avg_pool = nn.AvgPool3d(kernel_size=[1, 14, 14], stride=(1, 1, 1)) # reduce spatial downsize to 7x7
        self.feature_reduce = nn.Conv3d(512, 64, [1, 1, 1])
        self.name = 'r3d'
        
        if pretrained:
            model_kvpair = self.extractor.state_dict()
            for layer_name, weights in torch.load('./trained_models/R3D.pth')['state_dict'].items():
                layer_name = layer_name.replace('R3D_model.','')
                layer_name = layer_name.replace('extractor.','')
                if 'logits' in layer_name or 'actions' in layer_name or 'feature_dim' in layer_name or 'features' in layer_name or 'fc' in layer_name:
                    continue
                model_kvpair[layer_name]=weights
            self.extractor.load_state_dict(model_kvpair, strict=True)

       
    def forward(self, inputs):
        #print(inputs.shape)
        inputs = inputs.permute(0, 2, 1, 3, 4)
        features = self.extractor(inputs)
        features = list(features.values())
        features = features[0]
        #print(features.shape)
        #features = torch.squeeze(features) # b x f x t x h x w
        #features = self.feature_reduce(features)
        #print(features.shape)
        features = self.avg_pool(features) # b x t x f
        #print(features.shape)
        #print(features.shape)
        #features = features.permute(0, 2, 1, 3, 4)
        #print(features.shape)
        #features = features.flatten(2, 4) # b x t x (fhw) and reduce f to 64 using 1x1 conv and combine f and h and w - > b x t x (fhw)
        #print(features.shape)
        return features
        
        
class SwinBackbone(nn.Module):
    def __init__(self, pretrained, hidden_dim=512):
        super(SwinBackbone, self).__init__()
        weights = Swin_T_Weights.DEFAULT
        self.model = swin_t(weights=weights).cuda()
        #print(self.model)
        self.hidden_dim = hidden_dim
        #train, test = get_graph_node_names(model)
        #print(train, flush=True)
        #self.extractor = create_feature_extractor(self.model, return_nodes={"avgpool": "features"})
        self.avg_pool = nn.AvgPool3d(kernel_size=[1, 14, 14], stride=(1, 1, 1))
        
#        if pretrained:
#            model_kvpair = self.extractor.state_dict()
#            for layer_name, weights in torch.load('./trained_models/R3D.pth')['state_dict'].items():
#                layer_name = layer_name.replace('R3D_model.','')
#                layer_name = layer_name.replace('extractor.','')
#                if 'logits' in layer_name or 'actions' in layer_name or 'feature_dim' in layer_name or 'features' in layer_name or 'fc' in layer_name:
#                    continue
#                model_kvpair[layer_name]=weights
#            self.extractor.load_state_dict(model_kvpair, strict=True)

        
    def forward(self, inputs):
        frame_features = []
        #inputs = inputs.permute(0, 2, 1, 3, 4)
        print(inputs.shape)
        for batch in inputs:
            features = self.model(batch)
            frame_features.append(features)
        features = torch.stack([feature for feature in frame_features])
        return features


