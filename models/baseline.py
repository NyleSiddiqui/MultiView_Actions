import torch 
from torch import nn
from torchvision.models.video import r2plus1d_18, r3d_18, R2Plus1D_18_Weights, R3D_18_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


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

        
class R3DBackbone(nn.Module):
    def __init__(self):
        super(R3DBackbone, self).__init__()
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights).cuda()
        self.hidden_dim = 256
        self.R3D_model = model
        self.extractor = create_feature_extractor(model, return_nodes={"layer4": "features"}) #extract from earlier layer?
        self.avg_pool = nn.AvgPool3d(kernel_size=[1, 14, 14], stride=(1, 1, 1)) # reduce spatial downsize to 7x7
        self.name = 'r3d'

       
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        features = self.extractor(inputs)
        features = list(features.values())
        features = features[0]
        features = self.avg_pool(features) # b x t x f
        return features
        
        
