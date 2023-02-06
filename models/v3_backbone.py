import torch
from torch import nn
from models.baseline import R3DBackbone
from einops import repeat


class VideoTransformer(nn.Module):
    def __init__(self, input_size, num_frames, num_views, num_actions, patch_size, hidden_dim, num_heads, num_layers):
        super(VideoTransformer, self).__init__()
        self.input_size = input_size
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.pretrained = False
        self.blurred = False
        #self.backbone = VideoSwinBackbone(num_subjects, num_actions, True)
        self.backbone = R3DBackbone(pretrained=self.pretrained)
        #self.backbone = I3DBackbone(pretrained=True, num_subjects=num_subjects)
        #self.backbone = VideoSwinBackbone(num_subjects, num_actions, backbone=True)
        #self.backbone = gitviv(self.num_frames, self.num_subjects, self.num_actions, self.input_size, self.patch_size, num_heads=num_heads).cuda()

        if self.backbone.name == 'r3d':
            self.patch_size = num_frames // 8
            self.backbone_output = 512

        elif self.backbone.name == 'video_swin':
            self.patch_size = num_frames // 2
            self.backbone_output = 1024

        elif self.backbone.name == 'i3d':
            self.patch_size = num_frames // 4
            self.backbone_output = 1024
            
        elif self.backbone.name == 'vivit':
            self.backbone_output = 768
            self.patch_size = 1
            
            weights = True
            if weights:
                model_kvpair = self.backbone.state_dict()
                for layer_name, weights in torch.load("/home/siddiqui/Action_Biometrics/results/saved_models/mergedNTUPK+ViViT_12-01-23_1314/model_15_58.9525.pth")['state_dict'].items():
                    layer_name = layer_name.replace('model.','')
                    #layer_name = layer_name.replace('extractor.','')
                    if 'mlp' in layer_name or 'actions' in layer_name or 'feature_dim' in layer_name or 'features' in layer_name:
                        continue
                    model_kvpair[layer_name]=weights
                self.backbone.load_state_dict(model_kvpair, strict=True)
                print('weights', flush=True)
            
        self.positional_encoding = nn.Parameter(torch.zeros(self.patch_size, self.backbone_output), requires_grad=True) #torch zeroes?
        
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(self.backbone_output, num_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True, dropout=0.0, norm_first=False) for i in range(num_layers)])
        
        #self.subject_decoder = nn.ModuleList([nn.TransformerDecoderLayer(self.backbone_output, num_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True, dropout=0.0, norm_first=False) for i in range(num_layers)]) #feedforward?
        #self.action_decoder = nn.ModuleList([nn.TransformerDecoderLayer(self.backbone_output, num_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True, dropout=0.0, norm_first=False) for i in range(num_layers)])
        
        self.subject_decoder_l = nn.TransformerDecoderLayer(self.backbone_output, num_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True, dropout=0.0, norm_first=False)
        self.action_decoder_l = nn.TransformerDecoderLayer(self.backbone_output, num_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True, dropout=0.0, norm_first=False)
        self.subject_decoder = nn.TransformerDecoder(self.subject_decoder_l, self.num_layers)
        self.action_decoder = nn.TransformerDecoder(self.action_decoder_l, self.num_layers)
        
        self.action_tokens =  nn.Parameter(torch.unsqueeze(get_orthogonal_queries(20, self.backbone_output), dim=0))
        self.subject_tokens = nn.Parameter(torch.unsqueeze(get_orthogonal_queries(20, self.backbone_output), dim=0))
        
        self.subject_weights = nn.Parameter(torch.ones(self.num_views))
        self.action_weights = nn.Parameter(torch.ones(num_actions))
        
        self.mlp_head_subject = nn.Linear(self.backbone_output, self.num_views)
        self.mlp_head_action = nn.Linear(self.backbone_output, num_actions)
        
        
    def forward(self, inputs):
        bs = inputs.shape[0]
        if self.pretrained:
            with torch.no_grad():
                features = self.backbone(inputs)
        else:
            features = self.backbone(inputs)
            #print(features.shape)
        if self.backbone.name == 'vivit':
            #print(features.shape)
            features = features.unsqueeze(1)
            #print(features.shape)
        else:
            features = features.squeeze(-1).squeeze(-1)
            #print(features.shape)
            features = features.permute(0, 2, 1)
        features += self.positional_encoding[None, :features.shape[1], :]

        
        action_tokens = repeat(self.action_tokens, '() n d -> b n d', b=bs)
        subject_tokens = repeat(self.subject_tokens, '() n d -> b n d', b=bs)

            
        features_subject = self.subject_decoder(subject_tokens, features)
        features_action = self.action_decoder(action_tokens, features)
        
        
        #features_subject = torch.mean(features_subject * self.subject_weights[None, :, None], dim=1) # b x queries x f
        #features_action = torch.mean(features_action * self.action_weights[None, :, None], dim=1)
        
        features_subject = features_subject.mean(dim=1)
        features_action = features_action.mean(dim=1)
        
        output_subjects = self.mlp_head_subject(features_subject)
        output_actions = self.mlp_head_action(features_action)
        
        
        return output_subjects, output_actions, features_subject, features_action, subject_tokens, action_tokens
        
        
def generate_orthogonal_vectors(N,d):
    assert N >= d, "[generate_orthogonal_vectors] dim issue"
    init_vectors = torch.normal(0, 1, size=(N, d))
    norm_vectors = nn.functional.normalize(init_vectors, p=2.0, dim=1)

    # Compute the qr factorization
    q, r = torch.linalg.qr(norm_vectors)

    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    return q
    

def get_orthogonal_queries(n_classes, n_dim, apply_norm=True):
    if n_classes < n_dim:
        vecs = generate_orthogonal_vectors(n_dim, n_dim)[:n_classes, :]
    else:
        vecs = generate_orthogonal_vectors(n_dim, n_dim)

    if apply_norm:
        vecs = nn.functional.normalize(vecs, p=2.0, dim=1)

    return vecs