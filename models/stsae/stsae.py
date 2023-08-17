import torch.nn as nn
import torch
from typing import List, Tuple, Union

from models.common.components import Encoder, Decoder



class STSE(nn.Module):
    
    def __init__(self, c_in:int, h_dim:int=32, latent_dim:int=64, n_frames:int=12, 
                 n_joints:int=17, layer_channels:List[int]=[128, 64, 128], dropout:float=0.3, 
                 device:Union[str, torch.DeviceObjType]='cpu') -> None:
        """
        Space-Time-Separable Encoder (STSE).

        Args:
            c_in (int): number of coordinates of the input
            h_dim (int, optional): dimension of the hidden layer. Defaults to 32.
            latent_dim (int, optional): dimension of the latent space. Defaults to 64.
            n_frames (int, optional): number of frames of the input pose sequence. Defaults to 12.
            n_joints (int, optional): number of joints of the input pose sequence. Defaults to 17.
            layer_channels (List[int], optional): list of channel dimension for each layer. Defaults to [128, 64, 128].
            dropout (float, optional): dropout probability. Defaults to 0.3.
            device (Union[str, torch.DeviceObjType], optional): model device. Defaults to 'cpu'.
        """
        
        super(STSE, self).__init__()
        
        # Set the model's parameters
        self.input_dim = c_in
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.n_joints = n_joints
        self.layer_channels = layer_channels
        self.dropout = dropout
        self.device = device
        
        # Build the model
        self.__build_model()
        
    
    def build_model(self) -> None:
        """
        Build the model.
        """
        
        self.encoder = Encoder(input_dim=self.input_dim, layer_channels=self.layer_channels, 
                               hidden_dimension=self.h_dim, n_frames=self.n_frames, 
                               n_joints=self.n_joints, dropout=self.dropout)
        self.btlnk = nn.Linear(in_features=self.h_dim*self.n_frames*self.n_joints, 
                               out_features=self.latent_dim)
        
    
    __build_model = build_model
    
    
    def encode(self, X:torch.Tensor, return_shape:bool=False, t:torch.Tensor=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Size]]:
        """
        Encode the input pose sequence.
        
        Args:
            X (torch.Tensor): input pose sequence of shape (batch_size, n_frames, n_joints, input_dim)
            return_shape (bool, optional): whether to return the shape of the output tensor. Defaults to False.
            t (torch.Tensor, optional):conditioning signal for the STS-GCN layers. Defaults to None.
            
        Returns:
            torch.Tensor: latent representation of the input pose sequence of shape (batch_size, latent_dim)
        """
        
        assert len(X.shape) == 4
        X = X.unsqueeze(4)
        N, C, T, V, M = X.size()

        X = X.permute(0, 4, 3, 1, 2).contiguous()
        X = X.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
            
        # Encode the input pose sequence
        X, _ = self.encoder(X, t)
        N, C, T, V = X.size()
        X = X.view([N, -1]).contiguous()
        X = X.view(N, M, self.h_dim, T, V).permute(0, 2, 3, 4, 1).contiguous()
        X_shape = X.size()
        X = X.view(N, -1).contiguous()
        
        # Apply the bottleneck layer
        X = self.btlnk(X)
        
        if return_shape:
            return X, X_shape
        return X
        
    
    def forward(self, X:torch.Tensor, t:torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            X (torch.Tensor): input pose sequence of shape (batch_size, n_frames, n_joints, input_dim)
            t (torch.Tensor, optional):conditioning signal for the STS-GCN layers. Defaults to None.
            
        Returns:
            torch.Tensor: latent representation of the input pose sequence of shape (batch_size, latent_dim)
        """
        
        return self.encode(X, return_shape=False, t=t), None
    
    


class STSAE(STSE): 
    
    def __init__(self, c_in:int, h_dim:int=32, latent_dim:int=64, n_frames:int=12, 
                 n_joints:int=17, layer_channels:List[int]=[128, 64, 128], dropout:float=0.3, 
                 device:Union[str, torch.DeviceObjType]='cpu') -> None:
        """
        Space-Time-Separable Autoencoder (STSAE).

        Args:
            c_in (int): number of coordinates of the input
            h_dim (int, optional): dimension of the hidden layer. Defaults to 32.
            latent_dim (int, optional): dimension of the latent space. Defaults to 64.
            n_frames (int, optional): number of frames of the input pose sequence. Defaults to 12.
            n_joints (int, optional): number of joints of the input pose sequence. Defaults to 17.
            layer_channels (List[int], optional): list of channel dimension for each layer. Defaults to [128, 64, 128].
            dropout (float, optional): dropout probability. Defaults to 0.3.
            device (Union[str, torch.DeviceObjType], optional): model device. Defaults to 'cpu'.
        """
        
        super(STSAE, self).__init__(c_in, h_dim, latent_dim, n_frames, n_joints, layer_channels, dropout, device)
        
        # Build the model
        self.build_model()
        
        
    def build_model(self) -> None:
        """
        Build the model.
        """
        
        self.decoder = Decoder(output_dim=self.input_dim, layer_channels=self.layer_channels, 
                               hidden_dimension=self.h_dim, n_frames=self.n_frames, 
                               n_joints=self.n_joints, dropout=self.dropout)
        self.rev_btlnk = nn.Linear(in_features=self.latent_dim,
                                   out_features=self.h_dim*self.n_frames*self.n_joints)
    
    
    def decode(self, Z:torch.Tensor, input_shape:Tuple[int], t:torch.Tensor=None) -> torch.Tensor:
        """
        Decode the latent representation of the input pose sequence.

        Args:
            Z (torch.Tensor): latent representation of the input pose sequence of shape (batch_size, latent_dim)
            input_shape (Tuple[int]): shape of the input pose sequence
            t (torch.Tensor, optional): conditioning signal for the STS-GCN layers. Defaults to None.

        Returns:
            torch.Tensor: reconstructed pose sequence of shape (batch_size, input_dim, n_frames, n_joints)
        """
        
        Z = self.rev_btlnk(Z)
        N, C, T, V, M = input_shape
        Z = Z.view(input_shape).contiguous()
        Z = Z.permute(0, 4, 1, 2, 3).contiguous()
        Z = Z.view(N * M, C, T, V).contiguous()

        Z = self.decoder(Z)
        
        return Z
        
        
    def forward(self, X:torch.Tensor, t:torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            X (torch.Tensor): input pose sequence of shape (batch_size, input_dim, n_frames, n_joints)
            t (torch.Tensor, optional): conditioning signal for the STS-GCN layers. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: reconstructed pose sequence of shape (batch_size, input_dim, n_frames, n_joints)
            and latent representation of the input pose sequence of shape (batch_size, latent_dim)
        """
        
        hidden_X, X_shape = self.encode(X, return_shape=True)
        X = self.decode(hidden_X, X_shape, t)
        
        return hidden_X, X