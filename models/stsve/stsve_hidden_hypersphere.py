import torch.nn as nn
import torch.nn.functional as F
import torch

from power_spherical.distributions import PowerSpherical, HypersphericalUniform
from models.common.components import Encoder, Decoder



class STSVE(nn.Module):
    def __init__(self, c_in, h_dim=32, latent_dim=512, n_frames=12, n_joints=18, **kwargs) -> None:
        super(STSVE, self).__init__()
        
        dropout = kwargs.get('dropout', 0.3)
        channels = kwargs.get('channels', [128,64,128])
        decoder_channels = kwargs.get('decoder_channels', [8])
        projector = kwargs.get('projector', 'linear')
        
        self.encoder = Encoder(c_in, h_dim, n_frames, n_joints, dropout, channels)
        self.decoder = Decoder(c_out=c_in, h_dim=h_dim, n_frames=n_frames, n_joints=n_joints, dropout=dropout, channels=decoder_channels)
        
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.distribution = kwargs.get('distribution', 'normal')
        
        # code borrowed from https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py
        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean = nn.Linear(in_features=self.h_dim * n_frames * n_joints, out_features=self.latent_dim)
            self.fc_var =  nn.Linear(in_features=self.h_dim * n_frames * n_joints, out_features=self.latent_dim)
        elif self.distribution == 'ps':
            # compute mean and concentration of the PowerSpherical
            if projector=='linear':
                self.fc_mean = nn.Linear(in_features=self.h_dim * n_frames * n_joints, out_features=self.latent_dim)
                self.fc_var = nn.Linear(in_features=self.h_dim * n_frames * n_joints, out_features=1)
            else:
                self.btlnk = MLP(input_size = self.h_dim * n_frames * n_joints, hidden_size=[self.latent_dim,self.latent_dim])
                self.fc_mean = nn.Linear(in_features = self.latent_dim, out_features=self.latent_dim)
                self.fc_var = nn.Linear(in_features=self.latent_dim, out_features=1)
        else:
            raise NotImplemented
        
        self.rev_btlnk = nn.Linear(in_features=latent_dim, out_features=h_dim * n_frames * n_joints) # for the decoder
        
        self.register_buffer('mean_vector', torch.zeros((1, self.latent_dim))) # expected value of the distribution of the normal data
        self.register_buffer('threshold_dist', torch.tensor(0)) # threshold of the cosine distance 
        
        

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
        
        # borrowed from https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py
        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean = self.fc_mean(self.btlnk(x))
            z_var = F.softplus(self.fc_var(x)) + 1
        elif self.distribution == 'ps':
            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.fc_mean(self.btlnk(x))
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(self.btlnk(x))) + 1
        else:
            raise NotImplemented
        
        if return_shape:
            return z_mean, z_var, x_shape
        return z_mean, z_var
    
    
    def decode(self, z, input_shape):
        
        z = self.rev_btlnk(z)
        N, C, T, V, M = input_shape
        z = z.view(input_shape).contiguous()
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        z = z.view(N * M, C, T, V)

        z = self.decoder(z)
        
        return z
    
    
    # borrowed from https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py
    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'ps':
            q_z = PowerSpherical(loc=z_mean, scale=torch.squeeze(z_var, dim=-1))
            p_z = HypersphericalUniform(self.latent_dim - 1, device='cuda')
        else:
            raise NotImplemented

        return q_z, p_z
    
    
    def forward(self, x): 
        z_mean, z_var, input_shape = self.encode(x, return_shape=True)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x = self.decode(z, input_shape=input_shape)
        
        return z, x, (q_z, p_z, z_var)
    
    
class MLP(nn.Module):
    """MLP class for projector and predictor."""

    def __init__(self, input_size, hidden_size, hyper=False, bias=True):
        super().__init__()

        n_layers = len(hidden_size)
        layer_list = []

        for idx, next_dim in enumerate(hidden_size):

            if idx == n_layers-1:
                layer_list.append(nn.Linear(input_size, next_dim, bias=bias))
                # if bias is False:
                #     layer_list.append(nn.BatchNorm1d(next_dim, affine=False))
            else:
                # keep dim as in literature
                layer_list.append(nn.Linear(input_size, next_dim, bias=bias))
                layer_list.append(nn.BatchNorm1d(next_dim))
                layer_list.append(nn.ReLU(inplace=True))
                input_size = next_dim

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)
