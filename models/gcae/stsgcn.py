import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn



class ST_GCNN_layer(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int, kernel_size:Union[Tuple[int], List[int]],
                 stride:int, time_dim:int, joints_dim:int, dropout:float, bias:bool=True, emb_dim:int=None) -> None:
        """
        Space-Time-Seperable Graph Convolutional Layer.

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            kernel_size (Union[Tuple[int], List[int]]): kernel size of the convolutional layer
            stride (int): stride of the convolutional layer
            time_dim (int): time dimension
            joints_dim (int): joints dimension
            dropout (float): dropout probability
            bias (bool, optional): whether to use bias in the convolutional layers. Defaults to True.
            emb_dim (int, optional): embedding dimension. Defaults to None.
        """
        
        super(ST_GCNN_layer,self).__init__()
        
        # Set the model's parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.time_dim = time_dim
        self.joints_dim = joints_dim
        self.dropout = dropout
        self.bias = bias
        self.emb_dim = emb_dim
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        
        # Build the model
        self.build_model()

    
    def build_model(self) -> None:
        """
        Build the model.
        """
        
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        
        self.gcn=ConvTemporalGraphical(self.time_dim, self.joints_dim)
        
        self.tcn = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (self.stride, self.stride),
                padding,
                bias=self.bias
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.Dropout(self.dropout, inplace=True),
        )
        
        if self.stride != 1 or self.in_channels != self.out_channels: 

            self.residual = nn.Sequential(
                            nn.Conv2d(self.in_channels,
                                      self.out_channels,
                                      kernel_size=1,
                                      stride=(1, 1),
                                      bias=self.bias),
                            nn.BatchNorm2d(self.out_channels))   
            
        else:
            self.residual=nn.Identity()
    
        self.prelu = nn.PReLU()

        if self.emb_dim is not None:
            self.emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    self.emb_dim,
                    self.out_channels
                ),
            )
        

    def forward(self, X:torch.Tensor, t:torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]
            t (torch.Tensor, optional): time tensor of shape [batch_size, time_embedding_dim]. Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape [batch_size, out_channels, time_dim, joints_dim]
        """
        
        res = self.residual(X)
        X = self.gcn(X) 
        X = self.tcn(X)
        X = X + res
        X = self.prelu(X)
        
        if (self.emb_dim is not None) and (t is not None):
            emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, X.shape[-2], X.shape[-1]).contiguous()
            return X + emb
        else:
            return X.contiguous()



class ConvTemporalGraphical(nn.Module):
    
    def __init__(self, time_dim:int, joints_dim:int) -> None:
        """
        The basic module for applying a graph convolution.
        Source: https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
        
        Args:
            time_dim (int): number of frames
            joints_dim (int): number of joints
        """
        
        super(ConvTemporalGraphical,self).__init__()
        
        self.A=nn.Parameter(torch.FloatTensor(time_dim, joints_dim, joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.T=nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)
        
        
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]

        Returns:
            torch.Tensor: output tensor of shape [batch_size, in_channels, time_dim, joints_dim]
        """
        
        X = torch.einsum('nctv,vtq->ncqv', (X, self.T)).contiguous()
        X = torch.einsum('nctv,tvw->nctw', (X, self.A)).contiguous()
        return X.contiguous() 




class CNN_layer(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, kernel_size:Union[Tuple[int], List[int]],
                 dropout:float, bias=True) -> None:
        """
        This is the simple CNN layer that performs a 2-D convolution while maintaining the dimensions of the input (except for the features dimension).

        Args:
            in_channels (int): number of channels of the input
            out_channels (int): number of channels of the output
            kernel_size (Union[Tuple[int], List[int]]): kernel size of the convolution
            dropout (float): dropout probability
            bias (bool, optional): whether to use bias in the convolutional layers. Defaults to True.
        """
        
        super(CNN_layer,self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        self.block= [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
                     nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True)] 

        self.block=nn.Sequential(*self.block)
        

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]

        Returns:
            torch.Tensor: output tensor of shape [batch_size, out_channels, time_dim, joints_dim]
        """
        
        output= self.block(X)
        return output