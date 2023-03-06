import torch
from torch import nn
from models.msg3d import Model 
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
        self.backbone_output = 512
        self.model = Model(num_point=25, num_person=2, num_gcn_scales=13, num_g3d_scales=6, graph='graph.ntu_rgb_d.AdjMatrixGraph')
        self.linear = nn.Linear(384, self.backbone_output)
            
        self.positional_encoding = nn.Parameter(torch.zeros(self.patch_size, self.backbone_output), requires_grad=True) 
        
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(self.backbone_output, num_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True, dropout=0.0, norm_first=False) for i in range(num_layers)])
                    
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(self.backbone_output, num_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True, dropout=0.0, norm_first=False)
        self.tranformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers)
        
        self.action_tokens =  nn.Parameter(torch.unsqueeze(get_orthogonal_queries(20, self.backbone_output), dim=0))
        #self.view_tokens = nn.Parameter(torch.unsqueeze(get_orthogonal_queries(1, self.backbone_output), dim=0))
        self.view_tokens = nn.Parameter(torch.randn(1, 1, self.backbone_output))
        
        self.view_weights = nn.Parameter(torch.ones(self.num_views))
        self.action_weights = nn.Parameter(torch.ones(num_actions))
        
        self.mlp_head_view = nn.Linear(self.backbone_output, self.num_views)
        self.mlp_head_action = nn.Linear(self.backbone_output, num_actions)
        

    def forward(self, inputs):
        bs = inputs.shape[0]
    
        out = self.model(inputs).permute(0, 2, 1)
        features = self.linear(out)
    
        features += self.positional_encoding[None, :features.shape[1], :]
        #print(features.shape)
        
        action_tokens = repeat(self.action_tokens, '() n d -> b n d', b=bs)
        view_tokens = repeat(self.view_tokens, '() n d -> b n d', b=bs)
        #print(action_tokens.shape, view_tokens.shape)
        
        queries = torch.cat((action_tokens, view_tokens), dim=1) # b x 16 x f
        #print(queries.shape)
        out = self.tranformer_decoder(queries, features)
        #print(out.shape)
        x_act, x_view = out[:, :-1, :], out[:, -1, :]
        #print(x_act.shape, x_view.shape)
        
        
        #features_view = torch.mean(features_view * self.view_weights[None, :, None], dim=1) # b x queries x f
        #features_action = torch.mean(features_action * self.action_weights[None, :, None], dim=1)
        
        features_view = x_view
        features_action = x_act.mean(dim=1)
        
        output_views = self.mlp_head_view(features_view)
        output_actions = self.mlp_head_action(features_action)
        
        
        return output_views, output_actions, features_view, features_action, view_tokens, action_tokens
        
        
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