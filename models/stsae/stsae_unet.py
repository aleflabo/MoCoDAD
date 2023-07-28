from typing import List, Tuple, Union

import torch
import torch.nn as nn
from models.gcae.stsgcn import CNN_layer, ST_GCNN_layer


class STSE_Unet(nn.Module):
    
    # This dictionary is used to determine the number of joints to consider for each layer
    joints_to_consider = {'a': 17, 'b': 12, 'c': 10, 'd': 8}
    
    def __init__(self, c_in:int, embedding_dim:int=256, latent_dim:int=64, n_frames:int=12, n_joints:int=17,
                 unet_down_channels:List[int]=[16, 32, 32, 64, 64, 128, 6], 
                 dropout:float=0.3, device:Union[str, torch.DeviceObjType]='cpu',
                 set_out_layer:bool=True) -> None:
        """
        Class that downscales the input pose sequence along the joints dimension, expands the channels and
        (optionally) maps it onto the latent space.

        Args:
            c_in (int): number of coordinates of the input
            embedding_dim (int, optional): dimension of the the time embedding. Defaults to 256.
            latent_dim (int, optional): dimension of the latent space. Defaults to 64.
            n_frames (int, optional): number of frames of the input pose sequence. Defaults to 12.
            n_joints (int, optional): number of joints of the input pose sequence. Defaults to 17.
            unet_down_channels (List[int], optional): channels of the downscaling part of the Unet. Defaults to [16, 32, 32, 64, 64, 128, 6].
            dropout (float, optional): dropout probability. Defaults to 0.3.
            device (Union[str, torch.DeviceObjType], optional): model device. Defaults to 'cpu'.
            set_out_layer (bool, optional): set the output layer to map the input onto the latent space. Defaults to True.
        """
        
        super(STSE_Unet, self).__init__()
        
        # Set the model's main parameters
        self.input_dim = c_in
        self.embedding_dim = embedding_dim
        self.__latent_dim = latent_dim # Private attribute
        self.n_frames = n_frames
        self.n_joints = n_joints
        self.unet_down_channels = unet_down_channels
        self.dropout = dropout
        self.device = device
        
        # Build the model
        self.__build_model(set_out_layer)
        
        
    def build_model(self, set_out_layer:bool) -> None:
        """
        Build the model.

        Args:
            set_out_layer (bool): set the output layer to map the input onto the latent space
        """
        
        kernel_size = [1,1]
        stride = 1
        
        # Space-Time-Separable Graph Convolutional layers of the dowscaling part of the U-Net
        self.st_gcnnsp1a = nn.ModuleList()
        self.st_gcnnsp1a.append(
            ST_GCNN_layer(
                self.input_dim, self.unet_down_channels[0], kernel_size, stride, 
                self.n_frames, self.joints_to_consider['a'], 
                self.dropout, emb_dim=self.embedding_dim
            )
        )
        
        self.st_gcnnsd1 = nn.ModuleList()
        self.st_gcnnsd1.append(
            ST_GCNN_layer(
                self.unet_down_channels[0],
                self.unet_down_channels[1],
                kernel_size, stride,
                self.n_frames,
                self.joints_to_consider['a'],
                self.dropout,
                emb_dim=self.embedding_dim
            )
        )
        self.st_gcnnsd1.append(
            ST_GCNN_layer(
                self.unet_down_channels[1],
                self.unet_down_channels[2],
                kernel_size,
                stride,
                self.n_frames,
                self.joints_to_consider['a'],
                self.dropout,
                emb_dim=self.embedding_dim
            )
        )

        self.st_gcnnsd2 = nn.ModuleList()
        self.st_gcnnsd2.append(
            ST_GCNN_layer(
                self.unet_down_channels[2],
                self.unet_down_channels[3],
                kernel_size,
                stride,
                self.n_frames,
                self.joints_to_consider['b'],
                self.dropout,
                emb_dim=self.embedding_dim
            )
        )
        self.st_gcnnsd2.append(
            ST_GCNN_layer(
                self.unet_down_channels[3],
                self.unet_down_channels[4],
                kernel_size,
                stride,
                self.n_frames,
                self.joints_to_consider['b'],
                self.dropout,
                emb_dim=self.embedding_dim
            )
        )

        self.st_gcnnsd3 = nn.ModuleList()
        self.st_gcnnsd3.append(
            ST_GCNN_layer(
                self.unet_down_channels[4],
                self.unet_down_channels[5],
                kernel_size,
                stride,
                self.n_frames,
                self.joints_to_consider['c'],
                self.dropout,
                emb_dim=self.embedding_dim
            )
        )
        self.st_gcnnsd3.append(
            ST_GCNN_layer(
                self.unet_down_channels[5],
                self.unet_down_channels[6],
                kernel_size,
                stride,
                self.n_frames,
                self.joints_to_consider['c'],
                self.dropout,
                emb_dim=self.embedding_dim
            )
        )

        # Downscale along the joints dimension
        self.down1 = CNN_layer(self.joints_to_consider['a'], self.joints_to_consider['b'], kernel_size, self.dropout)
        self.down2 = CNN_layer(self.joints_to_consider['b'], self.joints_to_consider['c'], kernel_size, self.dropout)
        
        if set_out_layer:
            self.to_time_dim = torch.nn.Linear(in_features=self.unet_down_channels[6]*self.n_frames*self.joints_to_consider['c'], 
                                               out_features=self.__latent_dim)
    
    
    # Name mangling to save a private copy of the build_model method
    __build_model = build_model
        
        
    def pos_encoding(self, t:torch.Tensor, channels:int) -> torch.Tensor:
        """
        Positional encoding for embedding the time step.

        Args:
            t (torch.Tensor): time step
            channels (int): embedding dimension

        Returns:
            torch.Tensor: positional encoding
        """
        
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        ).to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def _downscale(self, X:torch.Tensor, t:torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Downscale the input pose sequence along the joints dimension.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, input_channels, n_frames, n_joints]
            t (torch.Tensor, optional): time embedding. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: downscaled tensor of shape [batch_size, expanded_channels, n_frames, scaled_n_joints], 
            residuals of the first and second downscaling
        """
        
        fd1 = X
        for gcn in self.st_gcnnsp1a:
            fd1 = gcn(fd1,t)

        # Apply the graph convolution on the coordinates dimension
        for gcn in self.st_gcnnsd1:
            fd1 = gcn(fd1,t)
            
        # Downscale the joints dimension
        d1 = fd1
        fd1 = self.down1(fd1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
           
        # Apply the graph convolution on the coordinates dimension
        for gcn in self.st_gcnnsd2:
            fd1 = gcn(fd1,t)

        # Downscale the joints dimension
        d2 = fd1
        fd1 = self.down2(fd1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
          
        # Apply the graph convolution on the coordinates dimension
        for gcn in self.st_gcnnsd3:
            fd1 = gcn(fd1,t)

        return fd1, d1, d2
    
        
    def forward(self, X:torch.Tensor, t:torch.Tensor) -> Tuple[torch.Tensor, List]:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, input_channels, n_frames, n_joints]
            t (torch.Tensor): time step

        Returns:
            Tuple[torch.Tensor, List]: output encoded sequence, list of the model's outputs (only for compatibility with the other models)
        """
        
        fd1, _, _ = self._downscale(X, t)
        
        fd1 = torch.flatten(fd1,1)
        fd1 = self.to_time_dim(fd1)
        
        return fd1, []
    
        
        
        
class STSAE_Unet(STSE_Unet):
    
    def __init__(self, c_in:int, embedding_dim:int=256, n_frames:int=12, n_joints:int=17,
                 unet_down_channels:List[int]=[16, 32, 32, 64, 64, 128, 64], 
                 unet_up_channels:List[int]=[64, 32, 32, 2], 
                 dropout:float=0.3, device:Union[str, torch.DeviceObjType]='cpu',
                 concat_condition:bool=True, inject_condition:bool=False) -> None:
        """
        Class that downscales the input pose sequence along the joints dimension, expands the channels and upscales it back.
        This class inherits from the STSE_Unet class, adding the upscaling logic to the parent class.

        Args:
            c_in (int): number of coordinates of the input
            embedding_dim (int, optional): dimension of the the time embedding. Defaults to 256.
            n_frames (int, optional): number of frames of the input pose sequence. Defaults to 12.
            n_joints (int, optional): number of joints of the input pose sequence. Defaults to 17.
            unet_down_channels (List[int], optional): channels of the downscaling part of the Unet. Defaults to [16, 32, 32, 64, 64, 128, 6].
            unet_up_channels (List[int], optional): _description_. Defaults to [64, 32, 32, 2].
            dropout (float, optional): dropout probability. Defaults to 0.3.
            device (Union[str, torch.DeviceObjType], optional): model device. Defaults to 'cpu'.
            concat_condition (bool, optional): concatenate the conditioning data to the input sequence. Defaults to True.
            inject_condition (bool, optional): provide the embedding of the conditioning data to the latent layers of the model. Defaults to False.
        """
        
        super(STSAE_Unet, self).__init__(c_in, embedding_dim, None, n_frames, n_joints, unet_down_channels,
                                         dropout, device, set_out_layer=False)
        
        # Set the model's main parameters (the other parameters are inherited from the parent class)
        self.unet_up_channels = unet_up_channels
        self.concat_condition = concat_condition
        self.inject_condition = inject_condition
        
        # Build the model
        self.build_model()
        
        
    def build_model(self, set_out_layer:bool=False) -> None:
        """
        Build the upscaling part of the model. The downscaling part is built by the parent class.
        """
        
        kernel_size = [1,1]
        stride = 1
        
        # Space-Time-Separable Graph Convolutional layers of the upscaling part of the U-Net
        self.st_gcnnsu4 = nn.ModuleList()
        self.st_gcnnsu4.append(
            ST_GCNN_layer(
                self.unet_down_channels[-1],
                self.unet_up_channels[0],
                kernel_size,
                stride,
                self.n_frames,
                self.joints_to_consider['b'],
                self.dropout,
                emb_dim=self.embedding_dim
            )
        )
        self.st_gcnnsu4.append(
            ST_GCNN_layer(
                self.unet_up_channels[0],
                self.unet_up_channels[1],
                kernel_size,
                stride,
                self.n_frames,
                self.joints_to_consider['b'],
                self.dropout,
                emb_dim=self.embedding_dim
            )
        )

        self.st_gcnnsu3 = nn.ModuleList()
        self.st_gcnnsu3.append(
            ST_GCNN_layer(
                self.unet_up_channels[1],
                self.unet_up_channels[2],
                kernel_size,
                stride,
                self.n_frames,
                self.joints_to_consider['a'],
                self.dropout,
                emb_dim=self.embedding_dim
            )
        )
        self.st_gcnnsu3.append(
            ST_GCNN_layer(
                self.unet_up_channels[2],
                self.unet_up_channels[3],
                kernel_size,
                stride,
                self.n_frames,
                self.joints_to_consider['a'],
                self.dropout,
                emb_dim=self.embedding_dim
            )
        )

        self.up2 = CNN_layer(self.joints_to_consider['b'], self.joints_to_consider['a'], kernel_size, self.dropout)
        self.up3 = CNN_layer(self.joints_to_consider['c'], self.joints_to_consider['b'], kernel_size, self.dropout)
    
    
    def _upscale(self, X:torch.Tensor, fd1:torch.Tensor, d1:torch.Tensor, d2:torch.Tensor, t:torch.Tensor=None) -> torch.Tensor:
        """
        Upscale the input pose sequence along the joints dimension.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, input_channels, n_frames, n_joints]
            fd1 (torch.Tensor): downscaled tensor of shape [batch_size, expanded_channels, n_frames, scaled_n_joints]
            d1 (torch.Tensor): residuals of the first downscaling
            d2 (torch.Tensor): residuals of the second downscaling
            t (torch.Tensor, optional): time embedding. Defaults to None.

        Returns:
            torch.Tensor: output sequence
        """
        
        # Upscale the joints dimension
        fd1 = self.up3(fd1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # Add residuals
        fd1 = fd1 + d2
        
        # Apply the graph convolution on the coordinates dimension
        for gcn in self.st_gcnnsu4:
            fd1 = gcn(fd1,t)

        # Upscale the joints dimension
        fd1 = self.up2(fd1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Add residuals
        fd1 = fd1 + d1
        
        # Apply the graph convolution on the coordinates dimension
        for gcn in self.st_gcnnsu3:
            fd1 = gcn(fd1,t)

        # Add residuals
        fd1 = fd1 + X

        return fd1
    
        
    def forward(self, X:torch.Tensor, t:torch.Tensor, pst:torch.Tensor=None, condition_data:torch.Tensor=None) -> Tuple[torch.Tensor, List]:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, input_channels, n_frames, n_joints]
            t (torch.Tensor): time step
            pst (torch.Tensor, optional): conditioning data. Defaults to None.
            condition_data (torch.Tensor, optional): conditioning data. Defaults to None.

        Returns:
            Tuple[torch.Tensor, List]: output sequence of shape [batch_size, input_channels, n_frames, n_joints], 
            list (only for compatibility with the other models)
        """
        
        # Encode the time step
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.embedding_dim)
        
        # Add conditioning signal
        if self.inject_condition:
            t = t + condition_data
        
        # Concatenate the conditioning signal
        if self.concat_condition:
            X = torch.cat([pst,X], dim=2)
        
        fd1, d1, d2 = self._downscale(X, t)

        fd1 = self._upscale(X, fd1, d1, d2, t)

        return fd1, []