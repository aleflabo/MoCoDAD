import torch.nn as nn

from models.gcae import stsgcn_diffusion_unet as stsgcn

import pdb

class Encoder(nn.Module):
    def __init__(self, c_in, h_dim, n_frames, n_joints, dropout, channels) -> None:
        super().__init__()
        
        self.model = nn.ModuleList()
        
        in_ch = c_in

        # for channel in channels:
        #     self.model.append(stsgcn.ST_GCNN_layer(in_ch,channel,[1,1],1,n_frames,
        #                                            n_joints,dropout))
        #     in_ch = channel
        
        self.enc_1 = stsgcn.ST_GCNN_layer(in_ch,channels[0],[1,1],1,n_frames,
                                                   n_joints,dropout)
        in_ch = channels[0]
        self.enc_2 = stsgcn.ST_GCNN_layer(in_ch,channels[1],[1,1],1,n_frames,
                                                   n_joints,dropout)
        in_ch = channels[1]
        self.enc_3 = stsgcn.ST_GCNN_layer(in_ch,channels[2],[1,1],1,n_frames,
                                                   n_joints,dropout)
        in_ch = channels[2]
            
        self.enc_4 = stsgcn.ST_GCNN_layer(in_ch,h_dim,[1,1],1,n_frames,
                                               n_joints,dropout)
        
        # self.model = nn.Sequential(*self.model)
        
    def forward(self, x, t):
        '''
        input shape: [BatchSize, in_Channels, n_frames, n_joints]
        output shape: [BatchSize, h_dim, n_frames, n_joints]
        '''
        res_0 = x
        res_1 = self.enc_1(x,t)
        res_2 = self.enc_2(res_1,t)
        res_3 = self.enc_3(res_2,t)
        x = self.enc_4(res_3,t)
        
        return x, [res_0, res_1, res_2, res_3]    
    
    
class Decoder(nn.Module):
    def __init__(self, c_out, h_dim, n_frames, n_joints, dropout, channels) -> None:
        super().__init__()
        
        self.model = nn.ModuleList()
        
        in_ch = h_dim

        # for channel in channels:
        #     self.model.append(stsgcn.ST_GCNN_layer(in_ch,channel,[1,1],1,n_frames,
        #                                            n_joints,dropout))
        #     in_ch = channel
            
            
        self.dec_1 = stsgcn.ST_GCNN_layer(in_ch,channels[2],[1,1],1,n_frames,
                                                   n_joints,dropout)
        in_ch = channels[2]
        self.dec_2 = stsgcn.ST_GCNN_layer(in_ch,channels[1],[1,1],1,n_frames,
                                                   n_joints,dropout)
        in_ch = channels[1]
        self.dec_3 = stsgcn.ST_GCNN_layer(in_ch,channels[0],[1,1],1,n_frames,
                                                   n_joints,dropout)
        in_ch = channels[0]
            
        self.dec_4 = stsgcn.ST_GCNN_layer(in_ch,c_out,[1,1],1,n_frames,
                                               n_joints,dropout)
            
            
        # self.out = nn.Linear(n_frames, int(n_frames/2))
        self.out = nn.Linear(n_frames, n_frames)
        # self.out = nn.Linear(n_frames, 3)
        # self.model.append(stsgcn.ST_GCNN_layer(in_ch,c_out,[1,1],1,n_frames,
        #                                        n_joints,dropout)) 
        
        # self.model = nn.Sequential(*self.model)
         

    def forward(self, x, t, res):
        '''
        input shape: [BatchSize, h_dim, n_frames, n_joints]
        output shape: [BatchSize, in_Channels, n_frames, n_joints]
        '''
        
        x = self.dec_1(x,t)+res[3]
        x = self.dec_2(x,t)+res[2]
        x = self.dec_3(x,t)+res[1]
        x = self.dec_4(x,t)+res[0]

        x = self.out(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        return x
