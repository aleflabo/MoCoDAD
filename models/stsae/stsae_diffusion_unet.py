import torch.nn as nn
import torch

from models.common.diffusion_components import Encoder, Decoder
from models.gcae.stsgcn_diffusion_unet import ST_GCNN_layer, CNN_layer


class STSAE(nn.Module):
    def __init__(self, c_in, h_dim=32, latent_dim=512, n_frames=12, n_joints=18, concat_condition=True, inject_condition=False, emb_dim=12, **kwargs) -> None:
        super(STSAE, self).__init__()
        
        dropout = kwargs.get('dropout', 0.3)
        channels = kwargs.get('channels', [128,64,128])
        all_frames = n_frames
        self.encoder = Encoder(c_in, h_dim, all_frames, n_joints, dropout, channels)
        self.decoder = Decoder(c_out=c_in, h_dim=h_dim, n_frames=all_frames, n_joints=n_joints, dropout=dropout, channels=channels)
        self.n_frames = n_frames
        self.btlnk = nn.Linear(in_features=h_dim * all_frames * n_joints, out_features=latent_dim)
        self.rev_btlnk = nn.Linear(in_features=latent_dim, out_features=h_dim * all_frames * n_joints)
        self.final = nn.Linear(in_features=h_dim * all_frames * n_joints, out_features=latent_dim)
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.concat_condition = concat_condition
        self.inject_condition = inject_condition
        # self.register_buffer('c', torch.zeros(self.latent_dim)) # center c 
        self.joints_to_consider={"a": 17, "b": 12, "c": 10, "d": 8}
        self.emb_dim = latent_dim
        print('embedding dimension of STSAE', self.emb_dim)
        self.device = kwargs.get('device', 'cpu')

        self.st_gcnnsp1a = nn.ModuleList()
        self.st_gcnnsp1a.append(
            ST_GCNN_layer(
                c_in, 16, [1, 1], 1, all_frames, 17, dropout, emb_dim=self.emb_dim
            )
        )

        
        self.st_gcnnsd1 = nn.ModuleList()
        self.st_gcnnsd1.append(
            ST_GCNN_layer(
                16,
                32,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["a"],
                dropout,
                emb_dim=self.emb_dim
            )
        )
        self.st_gcnnsd1.append(
            ST_GCNN_layer(
                32,
                32,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["a"],
                dropout,
                emb_dim=self.emb_dim
            )
        )

        ##d2 in 64 frames out->128 frames
        self.st_gcnnsd2 = nn.ModuleList()
        self.st_gcnnsd2.append(
            ST_GCNN_layer(
                32,
                64,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["b"],
                dropout,
                emb_dim=self.emb_dim
            )
        )
        self.st_gcnnsd2.append(
            ST_GCNN_layer(
                64,
                64,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["b"],
                dropout,
                emb_dim=self.emb_dim
            )
        )

        ##d3 in 128 frames out->256 frames
        self.st_gcnnsd3 = nn.ModuleList()
        self.st_gcnnsd3.append(
            ST_GCNN_layer(
                64,
                128,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["c"],
                dropout,
                emb_dim=self.emb_dim
            )
        )
        self.st_gcnnsd3.append(
            ST_GCNN_layer(
                128,
                64,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["c"],
                dropout,
                emb_dim=self.emb_dim
            )
        )

        # Descending using cnns
        self.down1 = CNN_layer(17, 12, [1, 1], dropout)
        self.down2 = CNN_layer(12, 10, [1, 1], dropout)

        # upscaling
        self.st_gcnnsu4 = nn.ModuleList()
        self.st_gcnnsu4.append(
            ST_GCNN_layer(
                64,
                64,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["b"],
                dropout,
                emb_dim=self.emb_dim
            )
        )
        self.st_gcnnsu4.append(
            ST_GCNN_layer(
                64,
                32,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["b"],
                dropout,
                emb_dim=self.emb_dim
            )
        )

        self.st_gcnnsu3 = nn.ModuleList()
        self.st_gcnnsu3.append(
            ST_GCNN_layer(
                32,
                32,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["a"],
                dropout,
                emb_dim=self.emb_dim
            )
        )
        self.st_gcnnsu3.append(
            ST_GCNN_layer(
                32,
                2,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["a"],
                dropout,
                emb_dim=self.emb_dim
            )
        )

        self.up2 = CNN_layer(12, 17, [1, 1], dropout)
        self.up3 = CNN_layer(10, 12, [1, 1], dropout)

        
        
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
        
    def forward(self, x, t, pst=None, condition_data=None):
        # Diffusion #####################
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.emb_dim)
        x_noise = x
        if self.inject_condition:
            t = t+condition_data
        if self.concat_condition:
            x = torch.cat([pst,x], dim=2)
        
        x_r = x
        res = x_r

        # hidden_x, res = self.encode(x, return_shape=False, t=t) # return the hidden representation of the data x
        # x = self.decode(hidden_x, [], res, t=t) #+ res

        # return x, hidden_x
        fd1 = x
        for gcn in self.st_gcnnsp1a:
            fd1 = gcn(fd1,t)

        # downscaling
        for gcn in self.st_gcnnsd1:
            fd1 = gcn(fd1,t)  # 64frames    36joints  4,64,10,36
            
        d1 = fd1  # 36->24 #64frames 4,64,10,36 ======> [batch, coord, frame, art]
        fd1 = self.down1(fd1.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )  # 36->24 #64frames
           
        for gcn in self.st_gcnnsd2:
            fd1 = gcn(fd1,t)  # 128 features 24joints

        d2 = fd1  # 128features 4,128,10,24
        fd1 = self.down2(fd1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # 24->14
          
        
        for gcn in self.st_gcnnsd3:
            fd1 = gcn(fd1,t)  # 128 features 14joints

        # upscaling
        fd1 = self.up3(fd1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # 14->24 #128frames
        fd1 = fd1 + d2
        for gcn in self.st_gcnnsu4:
            fd1 = gcn(fd1,t)  # 128frames    24joints  4,128,10,24
        fd1 = self.up2(fd1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # 24->36

        fd1 = fd1 + d1
        for gcn in self.st_gcnnsu3:
            fd1 = gcn(fd1,t)  # 64 features 36joints

        fd1 += x

        return fd1, []
    


###################################################################################
###################################################################################

class STSENC(nn.Module):
    def __init__(self, c_in, h_dim=32, latent_dim=512, n_frames=12, n_joints=18, emb_dim=12, **kwargs) -> None:
        super(STSENC, self).__init__()
        
        dropout = kwargs.get('dropout', 0.3)
        channels = kwargs.get('channels', [128,64,128])
        all_frames = n_frames
        self.encoder = Encoder(c_in, h_dim, all_frames, n_joints, dropout, channels)
        #self.decoder = Decoder(c_out=c_in, h_dim=h_dim, n_frames=all_frames, n_joints=n_joints, dropout=dropout, channels=channels)
        
        self.btlnk = nn.Linear(in_features=h_dim * all_frames * n_joints, out_features=latent_dim)
        self.rev_btlnk = nn.Linear(in_features=latent_dim, out_features=h_dim * all_frames * n_joints)
        self.final = nn.Linear(in_features=h_dim * all_frames * n_joints, out_features=latent_dim)
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        # self.register_buffer('c', torch.zeros(self.latent_dim)) # center c 
        self.joints_to_consider={"a": 17, "b": 12, "c": 10, "d": 8}
        
        
        self.device = kwargs.get('device', 'cpu')
        self.n_frames = n_frames
        self.st_gcnnsp1a = nn.ModuleList()
        self.st_gcnnsp1a.append(
            ST_GCNN_layer(
                c_in, 16, [1, 1], 1, all_frames, 17, dropout, emb_dim=self.emb_dim
            )
        )

        
        self.st_gcnnsd1 = nn.ModuleList()
        self.st_gcnnsd1.append(
            ST_GCNN_layer(
                16,
                32,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["a"],
                dropout,
                emb_dim=self.emb_dim
            )
        )
        self.st_gcnnsd1.append(
            ST_GCNN_layer(
                32,
                32,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["a"],
                dropout,
                emb_dim=self.emb_dim
            )
        )

        ##d2 in 64 frames out->128 frames
        self.st_gcnnsd2 = nn.ModuleList()
        self.st_gcnnsd2.append(
            ST_GCNN_layer(
                32,
                64,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["b"],
                dropout,
                emb_dim=self.emb_dim
            )
        )
        self.st_gcnnsd2.append(
            ST_GCNN_layer(
                64,
                64,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["b"],
                dropout,
                emb_dim=self.emb_dim
            )
        )

        ##d3 in 128 frames out->256 frames
        self.st_gcnnsd3 = nn.ModuleList()
        self.st_gcnnsd3.append(
            ST_GCNN_layer(
                64,
                128,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["c"],
                dropout,
                emb_dim=self.emb_dim
            )
        )
        self.st_gcnnsd3.append(
            ST_GCNN_layer(
                128,
                6,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["c"],
                dropout,
                emb_dim=self.emb_dim
            )
        )

        # Descending using cnns
        self.down1 = CNN_layer(17, 12, [1, 1], dropout)
        self.down2 = CNN_layer(12, 10, [1, 1], dropout)

        # upscaling
        self.st_gcnnsu4 = nn.ModuleList()
        self.st_gcnnsu4.append(
            ST_GCNN_layer(
                64,
                64,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["b"],
                dropout,
                emb_dim=self.emb_dim
            )
        )
        self.st_gcnnsu4.append(
            ST_GCNN_layer(
                64,
                32,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["b"],
                dropout,
                emb_dim=self.emb_dim
            )
        )

        self.st_gcnnsu3 = nn.ModuleList()
        self.st_gcnnsu3.append(
            ST_GCNN_layer(
                32,
                32,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["a"],
                dropout,
                emb_dim=self.emb_dim
            )
        )
        self.st_gcnnsu3.append(
            ST_GCNN_layer(
                32,
                2,
                [1, 1],
                1,
                all_frames,
                self.joints_to_consider["a"],
                dropout,
                emb_dim=self.emb_dim
            )
        )
        self.to_time_dim = torch.nn.Linear(in_features=6*self.n_frames*10, out_features=self.latent_dim) #remove hardcoded values
        
        #self.up2 = CNN_layer(12, 17, [1, 1], dropout)
        #self.up3 = CNN_layer(10, 12, [1, 1], dropout)

        
        
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
        #t = t.unsqueeze(-1).type(torch.float)
        #t = self.pos_encoding(t, self.emb_dim)
        
        condition = x

        #x = torch.cat([pst,x], dim=2)
        
        x_r = x
        res = x_r

        # hidden_x, res = self.encode(x, return_shape=False, t=t) # return the hidden representation of the data x
        # x = self.decode(hidden_x, [], res, t=t) #+ res

        # return x, hidden_x
        fd1 = x
        for gcn in self.st_gcnnsp1a:
            fd1 = gcn(fd1,t)

        # downscaling
        for gcn in self.st_gcnnsd1:
            fd1 = gcn(fd1,t)  # 64frames    36joints  4,64,10,36
            
        d1 = fd1  # 36->24 #64frames 4,64,10,36 ======> [batch, coord, frame, art]
        fd1 = self.down1(fd1.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )  # 36->24 #64frames
           
        for gcn in self.st_gcnnsd2:
            fd1 = gcn(fd1,t)  # 128 features 24joints

        d2 = fd1  # 128features 4,128,10,24
        fd1 = self.down2(fd1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # 24->14
          
        
        for gcn in self.st_gcnnsd3:
            fd1 = gcn(fd1,t)  # 128 features 14joints

        fd1 = torch.flatten(fd1,1)
        fd1 = self.to_time_dim(fd1)
        
        return fd1, []
    


