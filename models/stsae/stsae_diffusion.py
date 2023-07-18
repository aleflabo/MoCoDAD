import torch.nn as nn
import torch

from models.common.diffusion_components import Encoder, Decoder
from models.gcae.stsgcn_diffusion_unet import ST_GCNN_layer, CNN_layer

class STSAE(nn.Module):
    def __init__(self, c_in, h_dim=32, latent_dim=512, n_frames=12, n_joints=18, **kwargs) -> None:
        super(STSAE, self).__init__()
        
        dropout = kwargs.get('dropout', 0.3)
        channels = kwargs.get('channels', [128,64,128])
        all_frames = n_frames
        self.encoder = Encoder(c_in, h_dim, all_frames, n_joints, dropout, channels)
        self.decoder = Decoder(c_out=c_in, h_dim=h_dim, n_frames=all_frames, n_joints=n_joints, dropout=dropout, channels=channels)
        
        self.btlnk = nn.Linear(in_features=h_dim * all_frames * n_joints, out_features=latent_dim)
        self.rev_btlnk = nn.Linear(in_features=latent_dim, out_features=h_dim * all_frames * n_joints)
        self.final = nn.Linear(in_features=h_dim * all_frames * n_joints, out_features=latent_dim)
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        # self.register_buffer('c', torch.zeros(self.latent_dim)) # center c 
        self.emb_dim = 12
        self.device = kwargs.get('device', 'cpu')

        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def encode(self, x, return_shape=False, t=None):
        assert len(x.shape) == 4
        x = x.unsqueeze(4)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
            
        x, res = self.encoder(x, t)
        # N, C, T, V = x.shape
        # x = x.view([N, -1]).contiguous()
        # x = x.view(N, M, self.h_dim, T, V).permute(0, 2, 3, 4, 1)
        x_shape = x.size()
        # x = x.view(N, -1) 
        # x = self.btlnk(x)
        
        if return_shape:
            return x, x_shape, res
        return x, res
    
    def decode(self, z, input_shape, res, t=None):
        
        # z = self.rev_btlnk(z)
        # N, C, T, V, M = input_shape
        # z = z.view(input_shape).contiguous()
        # z = z.permute(0, 4, 1, 2, 3).contiguous()
        # z = z.view(N * M, C, T, V)

        z = self.decoder(z, t, res)
        
        return z
        
    def forward(self, x, t, pst=None):
        # Diffusion #####################
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.emb_dim)
        
        x_noise = x

        x = torch.cat([pst,x], dim=2)
        
        x_r = x
        res = x_r

        hidden_x, res = self.encode(x, return_shape=False, t=t) # return the hidden representation of the data x
        x = self.decode(hidden_x, [], res, t=t) #+ res

        return x, hidden_x
    
    
