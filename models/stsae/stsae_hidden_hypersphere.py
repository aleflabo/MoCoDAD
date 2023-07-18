import torch.nn as nn
import torch

from models.common.components import Encoder, Decoder



class STSAE(nn.Module): #used if self.ae 
    def __init__(self, c_in, h_dim=32, latent_dim=512, n_frames=12, n_joints=18, **kwargs) -> None:
        super(STSAE, self).__init__()
        
        dropout = kwargs.get('dropout', 0.3)
        channels = kwargs.get('channels', [128,64,128])
        self.encoder = Encoder(c_in, h_dim, n_frames, n_joints, dropout, channels)
        self.decoder = Decoder(c_out=c_in, h_dim=h_dim, n_frames=n_frames, n_joints=n_joints, dropout=dropout, channels=channels)
        
        self.btlnk = nn.Linear(in_features=h_dim * n_frames * n_joints, out_features=latent_dim)
        self.rev_btlnk = nn.Linear(in_features=latent_dim, out_features=h_dim * n_frames * n_joints)
        self.h_dim = h_dim
        self.n_joints = n_joints
        self.latent_dim = latent_dim
        self.register_buffer('c', torch.zeros(self.latent_dim)) # center c 
        
    def pos_encoding(self, t, channels):
        channels = channels+1
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)[:,:self.n_joints]
        return pos_enc

    def encode(self, x, return_shape=False):
        assert len(x.shape) == 4
        x = x.unsqueeze(4)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
            
        x = self.encoder(x)
        N, C, T, V = x.shape
        x = x.view([N, -1]).contiguous()
        x = x.view(N, M, self.h_dim, T, V).permute(0, 2, 3, 4, 1)
        x_shape = x.size()
        x = x.view(N, -1) 
        x = self.btlnk(x)
        
        if return_shape:
            return x, x_shape
        return x
    
    def decode(self, z, input_shape):
        
        z = self.rev_btlnk(z)
        N, C, T, V, M = input_shape
        z = z.view(input_shape).contiguous()
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        z = z.view(N * M, C, T, V)

        z = self.decoder(z)
        
        return z
        
    def forward(self, x):
        hidden_x, x_shape = self.encode(x, return_shape=True) # return the hidden representation of the data x
        x = self.decode(hidden_x, x_shape)
        
        return x, hidden_x
    
    # def forward(self, x, indices=None):

        

    #     """ first = True
    #     if indices is not None:
    #         for ind in indices:
    #             t = ind.unsqueeze(-1).type(torch.float)
    #             t = self.pos_encoding(t,x.shape[-1])
    #             if first:
    #                 emb = t.unsqueeze(0)
    #                 first = False
    #             else:
    #                 emb = torch.cat((emb,t.unsqueeze(0)),0)
    #         emb = emb.unsqueeze(1).repeat(1,2,1,1)
    #         x = x+emb"""
        
    #     t = indices.reshape(-1)
    #     t = t.unsqueeze(-1).type(torch.float)
    #     t = self.pos_encoding(t,x.shape[-1])
    #     t = t.reshape(indices.shape[0],indices.shape[1],x.shape[-1]).unsqueeze(1).repeat(1,x.shape[1],1,1)
    #     x = x+t
    #     hidden_x, x_shape = self.encode(x, return_shape=True) # return the hidden representation of the data x
    #     x = self.decode(hidden_x, x_shape)
        
    #     return x, hidden_x