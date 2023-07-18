import torch.nn as nn

from models.gcae import stsgcn



class Encoder(nn.Module):
    def __init__(self, c_in, h_dim, n_frames, n_joints, dropout, channels) -> None:
        super().__init__()
        
        self.model = nn.ModuleList()
        
        in_ch = c_in
        
        for channel in channels:
            self.model.append(stsgcn.ST_GCNN_layer(in_ch,channel,[1,1],1,n_frames,
                                                   n_joints,dropout))
            in_ch = channel
            
        self.model.append(stsgcn.ST_GCNN_layer(in_ch,h_dim,[1,1],1,n_frames,
                                               n_joints,dropout)) 
        
        self.model = nn.Sequential(*self.model)
        
    def forward(self, x):
        '''
        input shape: [BatchSize, in_Channels, n_frames, n_joints]
        output shape: [BatchSize, h_dim, n_frames, n_joints]
        '''
        return self.model(x)
    
    
    
class Decoder(nn.Module):
    def __init__(self, c_out, h_dim, n_frames, n_joints, dropout, channels) -> None:
        super().__init__()
        
        self.model = nn.ModuleList()
        
        in_ch = h_dim
        
        for channel in channels:
            self.model.append(stsgcn.ST_GCNN_layer(in_ch,channel,[1,1],1,n_frames,
                                                   n_joints,dropout))
            in_ch = channel
            
        self.model.append(stsgcn.ST_GCNN_layer(in_ch,c_out,[1,1],1,n_frames,
                                               n_joints,dropout)) 
        
        self.model = nn.Sequential(*self.model)
         

    def forward(self, x):
        '''
        input shape: [BatchSize, h_dim, n_frames, n_joints]
        output shape: [BatchSize, in_Channels, n_frames, n_joints]
        '''
        return self.model(x)