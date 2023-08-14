from typing import List, Tuple, Union

import torch
import torch.nn as nn
import models.gcae.stsgcn as stsgcn


class Encoder(nn.Module):
    
    def __init__(self, input_dim:int, layer_channels:List[int], hidden_dimension:int, 
                 n_frames:int, n_joints:int, dropout:float,
                 bias=True) -> None:
        """
        Class that implements a Space-Time-Separable Graph Convolutional Encoder (STS-GCN).

        Args:
            input_dim (int): number of coordinates of the input
            layer_channels (List[int]): list of channel dimension for each layer
            hidden_dimension (int): dimension of the hidden layer
            n_frames (int): number of frames of the input pose sequence
            n_joints (int): number of joints of the input pose sequence
            dropout (float): dropout probability
            bias (bool, optional): whether to use bias in the convolutional layers. Defaults to True.
        """
        
        super().__init__()
        
        # Set the model's parameters
        self.input_dim = input_dim
        self.layer_channels = layer_channels
        self.hidden_dimension = hidden_dimension
        self.n_frames = n_frames
        self.n_joints = n_joints
        self.dropout = dropout
        self.bias = bias
        
        # Build the model
        self.build_model()
        

    def build_model(self):
        """
        Build the model.

        Returns:
            nn.ModuleList: list of the model's layers
        """
        
        input_channels = self.input_dim
        layer_channels = self.layer_channels + [self.hidden_dimension]
        kernel_size = [1,1]
        stride = 1
        model_layers = nn.ModuleList()
        for channels in layer_channels:
            model_layers.append(
                stsgcn.ST_GCNN_layer(in_channels=input_channels, 
                                     out_channels=channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     time_dim=self.n_frames,
                                     joints_dim=self.n_joints,
                                     dropout=self.dropout,
                                     bias=self.bias))
            input_channels = channels
        self.model_layers = model_layers
        
        
    def forward(self, X:torch.Tensor, t:torch.Tensor=None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, input_channels, n_frames, n_joints]
            t (torch.Tensor): time tensor of shape [batch_size, n_frames]. Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape [batch_size, hidden_dimension, n_frames, n_joints]
            List[torch.Tensor]: list of the output tensors of each intermediate layer
        """
        
        layers_out = [X]
        for layer in self.model_layers:
            out_X = layer(layers_out[-1], t)
            layers_out.append(out_X)
        
        return layers_out[-1], layers_out[:-1]
    
    
 
    
class Decoder(nn.Module):
    
    def __init__(self, output_dim:int, layer_channels:List[int], hidden_dimension:int, 
                 n_frames:int, n_joints:int, dropout:float,
                 bias=True) -> None:
        """
        Class that implements a Space-Time-Separable Graph Convolutional Decoder (STS-GCN).

        Args:
            output_dim (int): number of coordinates of the output
            layer_channels (List[int]): list of channel dimension for each layer (in the same order as the encoder's layers)
            hidden_dimension (int): dimension of the hidden layer
            n_frames (int): number of frames of the input pose sequence
            n_joints (int): number of joints of the input pose sequence
            dropout (float): dropout probability
            bias (bool, optional): whether to use bias in the convolutional layers. Defaults to True.
        """
        
        super().__init__()
        
        # Set the model's parameters
        self.output_dim = output_dim
        self.layer_channels = layer_channels[::-1]
        self.hidden_dimension = hidden_dimension
        self.n_frames = n_frames
        self.n_joints = n_joints
        self.dropout = dropout
        self.bias = bias
        
        # Build the model
        self.build_model()
        
    
    def build_model(self):
        """
        Build the model.
        """
        
        input_channels = self.hidden_dimension
        layer_channels = self.layer_channels + [self.output_dim]
        kernel_size = [1,1]
        stride = 1
        model_layers = nn.ModuleList()
        for channels in layer_channels:
            model_layers.append(
                stsgcn.ST_GCNN_layer(in_channels=input_channels, 
                                     out_channels=channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     time_dim=self.n_frames,
                                     joints_dim=self.n_joints,
                                     dropout=self.dropout,
                                     bias=self.bias))
            input_channels = channels
        
        self.model_layers = model_layers
         

    def forward(self, X:torch.Tensor, t:torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, hidden_dimension, n_frames, n_joints]
            t (torch.Tensor): time tensor of shape [batch_size, n_frames]. Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape [batch_size, output_dim, n_frames, n_joints]
        """
        
        for layer in self.model_layers:
            X = layer(X, t)
        
        return X
            
            
            
        
class DecoderResiduals(Decoder):
    
    def build_model(self) -> None:
        """
        Build the model.
        """
        
        super().build_model()
        self.out = nn.Linear(self.n_frames, self.n_frames)
    
    
    def forward(self, X:torch.Tensor, t:torch.Tensor, residuals:List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, hidden_dimension, n_frames, n_joints]
            t (torch.Tensor): time tensor of shape [batch_size, n_frames]
            residuals (List[torch.Tensor]): list of the output tensors of each intermediate layer

        Returns:
            torch.Tensor: output tensor of shape [batch_size, output_dim, n_frames, n_joints]
        """
        
        for layer in self.model_layers:
            out_X = layer(X, t)
            X = out_X + residuals.pop()

        X = self.out(X.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2).contiguous()

        return X
    


class Denoiser(nn.Module):
    def __init__(self, input_size:int, hidden_sizes:List[int], cond_size:int=None, bias:bool=True, device:Union[str, torch.DeviceObjType]='cpu') -> None:
        """
        Class that implements a denoiser network for diffusion in the latent space.

        Args:
            input_size (int): size of the input
            hidden_sizes (List[int]): list of hidden sizes
            cond_size (int, optional): size of the conditioning embedding. Defaults to None.
            bias (bool, optional): add bias. Defaults to True.
            device (Union[str, torch.DeviceObjType], optional): device to use. Defaults to 'cpu'.
        """
        
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.cond_size = cond_size
        self.embedding_dim = self.cond_size
        self.bias = bias
        self.device = device
        
        # Build the model
        self.build_model()
        
        
    def build_model(self) -> None:
        self.net = nn.ModuleList()
        self.cond_layers = nn.ModuleList() if self.cond_size is not None else None
        n_layers = len(self.hidden_sizes)
        input_size = self.input_size
        for idx, next_dim in enumerate(self.hidden_sizes):
            if self.cond_size is not None:
                self.cond_layers.append(nn.Linear(self.cond_size, next_dim, bias=self.bias))
            if idx == n_layers-1:
                self.net.append(nn.Linear(input_size, next_dim, bias=self.bias))
            else:
                self.net.append(nn.Sequential(nn.Linear(input_size, next_dim, bias=self.bias),
                                              nn.BatchNorm1d(next_dim), nn.ReLU(inplace=True)))
                input_size = next_dim
                
                
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
                

    def forward(self, X:torch.Tensor, t:torch.Tensor, cond:torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, input_size]
            t (torch.Tensor): time tensor of shape [batch_size]
            cond (torch.Tensor, optional): input tensor of shape [batch_size, cond_size]. Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape [batch_size, hidden_sizes[-1]]
        """
        # Encode the time step
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.embedding_dim)
        
        # Add conditioning signal
        if cond is not None:
            cond = t + cond
        else:
            cond = t
        
        for i in range(len(self.net)):
            X = self.net[i](X)
            if cond is not None:
                X = X + self.cond_layers[i](cond)
        return X