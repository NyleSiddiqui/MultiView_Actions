import torch
from torch import nn
from models.baseline import R3DBackbone
from einops import repeat


class VideoTransformer(nn.Module):
    def __init__(self, input_size, num_views, num_actions):
        super(VideoTransformer, self).__init__()
        self.input_size = input_size
        self.num_views = num_views
        self.num_actions = num_actions
        self.backbone = R3DBackbone()

        if self.backbone.name == 'r3d':
            self.patch_size =  (16 // 8) 
            self.backbone_output = 512

            
        self.positional_encoding = nn.Parameter(torch.zeros(16, self.backbone_output), requires_grad=True) 
                            
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(self.backbone_output, 8, dim_feedforward=256, activation='gelu', batch_first=True, dropout=0.0, norm_first=False)
        self.tranformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, 2)
        
        num_queries = 30
        #print(f'num queries: {num_queries}', flush=True)
        self.action_tokens =  nn.Parameter(torch.unsqueeze(get_orthogonal_queries(num_queries, self.backbone_output), dim=0))
        num_viewqueries = 1
        self.view_tokens = nn.Parameter(torch.randn(1, num_viewqueries, self.backbone_output))
        
        self.mlp_head_view = nn.Linear(self.backbone_output, self.num_views)
        self.mlp_head_action = nn.Linear(self.backbone_output, num_actions)
        

    def forward(self, inputs):
        bs = inputs.shape[0]
    
        features = self.backbone(inputs)
        features = features.squeeze(-1).squeeze(-1)
        features = features.permute(0, 2, 1)
        features += self.positional_encoding[None, :features.shape[1], :]
        
        action_tokens = repeat(self.action_tokens, '() n d -> b n d', b=bs)
        view_tokens = repeat(self.view_tokens, '() n d -> b n d', b=bs)
        
        queries = torch.cat((action_tokens, view_tokens), dim=1) # b x 16 x f
        out = self.tranformer_decoder(queries, features)
        x_act, x_view = out[:, :-1, :], out[:, -1, :]
        
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
