from typing import List, Tuple

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
        self.model_layers = self.build_model()
        
    
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
            out_X = layer(X, t)
            X = out_X
            
            
            
        
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

        X = self.out(X.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        return X