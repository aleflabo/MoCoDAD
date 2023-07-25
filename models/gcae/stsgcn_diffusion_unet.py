#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import math




class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 time_dim,
                 joints_dim
    ):
        super(ConvTemporalGraphical,self).__init__()
        
        self.A=nn.Parameter(torch.FloatTensor(time_dim, joints_dim,joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv,stdv)

        self.T=nn.Parameter(torch.FloatTensor(joints_dim , time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv,stdv)
        '''
        self.prelu = nn.PReLU()
        
        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''
    def forward(self, x):
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        ## x=self.prelu(x)
        x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        return x.contiguous() 




class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True, 
                 emb_dim=256):
        
        super(ST_GCNN_layer,self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        
        self.gcn=ConvTemporalGraphical(time_dim,joints_dim) # the convolution layer
        
        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        
        if stride != 1 or in_channels != out_channels: 

            self.residual = nn.Sequential(
                            nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=1,
                                      stride=(1, 1),
                                      bias=bias),
                            nn.BatchNorm2d(out_channels))   
            
        else:
            self.residual=nn.Identity()
        
        
        self.prelu = nn.PReLU()

        if emb_dim is not None:
            self.emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_dim,
                    out_channels
                ),
            )
            
        self.emb_dim = emb_dim

        

    def forward(self, x, t=None):
        res=self.residual(x)
        x=self.gcn(x) 
        x=self.tcn(x)
        x=x+res
        x=self.prelu(x)
        
        if self.emb_dim is not None and t is not None:
            emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            return x+emb
        else:
            return x




class CNN_layer(nn.Module): # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
        
        super(CNN_layer,self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        
        
        self.block= [nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
                     ,nn.BatchNorm2d(out_channels),nn.Dropout(dropout, inplace=True)] 



            
        
        self.block=nn.Sequential(*self.block)
        

    def forward(self, x):
        
        output= self.block(x)
        return output


class Model(nn.Module):
    """ 
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_channels,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """

    def __init__(self,
                 input_channels,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 joints_to_consider,
                 n_txcnn_layers,
                 txc_kernel_size,
                 txc_dropout,
                 bias=True):
        
        super(Model,self).__init__()
        self.input_time_frame=input_time_frame
        self.output_time_frame=output_time_frame
        self.joints_to_consider={"a": 36, "b": 24, "c": 14, "d": 8}
        self.st_gcnnsd1 = nn.ModuleList()
        self.n_txcnn_layers=n_txcnn_layers
        self.txcnns1 = nn.ModuleList()
        
      
        self.st_gcnnsd1.append(
            ST_GCNN_layer(
                16,
                32,
                [1, 1],
                1,
                input_time_frame,
                self.joints_to_consider["a"],
                st_gcnn_dropout,
            )
        )
        self.st_gcnnsd1.append(
            ST_GCNN_layer(
                32,
                32,
                [1, 1],
                1,
                input_time_frame,
                self.joints_to_consider["a"],
                st_gcnn_dropout,
            )
        )

        # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)           
        self.txcnns.append(CNN_layer(input_time_frame,output_time_frame,txc_kernel_size,txc_dropout)) # with kernel_size[3,3] the dimensinons of C,V will be maintained       
        for i in range(1,n_txcnn_layers):
            self.txcnns.append(CNN_layer(output_time_frame,output_time_frame,txc_kernel_size,txc_dropout))
        
            
        self.prelus = nn.ModuleList()
        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())


        ##d2 in 64 frames out->128 frames
        self.st_gcnnsd2 = nn.ModuleList()
        self.st_gcnnsd2.append(
            ST_GCNN_layer(
                32,
                64,
                [1, 1],
                1,
                input_time_frame,
                self.joints_to_consider["b"],
                st_gcnn_dropout,
            )
        )
        self.st_gcnnsd2.append(
            ST_GCNN_layer(
                64,
                64,
                [1, 1],
                1,
                input_time_frame,
                self.joints_to_consider["b"],
                st_gcnn_dropout,
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
                input_time_frame,
                self.joints_to_consider["c"],
                st_gcnn_dropout,
            )
        )
        self.st_gcnnsd3.append(
            ST_GCNN_layer(
                128,
                64,
                [1, 1],
                1,
                input_time_frame,
                self.joints_to_consider["c"],
                st_gcnn_dropout,
            )
        )

        # Descending using cnns
        self.down1 = CNN_layer(36, 24, [1, 1], txc_dropout)
        self.down2 = CNN_layer(24, 14, [1, 1], txc_dropout)

        # upscaling
        self.st_gcnnsu4 = nn.ModuleList()
        self.st_gcnnsu4.append(
            ST_GCNN_layer(
                64,
                64,
                [1, 1],
                1,
                input_time_frame,
                self.joints_to_consider["b"],
                st_gcnn_dropout,
            )
        )
        self.st_gcnnsu4.append(
            ST_GCNN_layer(
                64,
                32,
                [1, 1],
                1,
                input_time_frame,
                self.joints_to_consider["b"],
                st_gcnn_dropout,
            )
        )

        self.st_gcnnsu3 = nn.ModuleList()
        self.st_gcnnsu3.append(
            ST_GCNN_layer(
                32,
                32,
                [1, 1],
                1,
                input_time_frame,
                self.joints_to_consider["a"],
                st_gcnn_dropout,
            )
        )
        self.st_gcnnsu3.append(
            ST_GCNN_layer(
                32,
                3,
                [1, 1],
                1,
                input_time_frame,
                self.joints_to_consider["a"],
                st_gcnn_dropout,
            )
        )

        self.up2 = CNN_layer(24, 36, [1, 1], txc_dropout)
        self.up3 = CNN_layer(14, 24, [1, 1], txc_dropout)


        

    def forward(self, x):

        # downscaling
        for gcn in self.st_gcnnsd1:
            fd1 = gcn(x)  # 64frames    36joints  4,64,10,36
            
        d1 = fd1  # 36->24 #64frames 4,64,10,36
        fd1 = self.down1(fd1.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )  # 36->24 #64frames
           
        for gcn in self.st_gcnnsd2:
            fd1 = gcn(fd1)  # 128 features 24joints

        d2 = fd1  # 128features 4,128,10,24
        fd1 = self.down2(fd1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # 24->14
          
        
        for gcn in self.st_gcnnsd3:
            fd1 = gcn(fd1)  # 128 features 14joints

        # upscaling
        fd1 = self.up3(fd1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # 14->24 #128frames
        fd1 = fd1 + d2
        for gcn in self.st_gcnnsu4:
            fd1 = gcn(fd1)  # 128frames    24joints  4,128,10,24
        fd1 = self.up2(fd1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # 24->36

        fd1 = fd1 + d1
        for gcn in self.st_gcnnsu3:
            fd1 = gcn(fd1)  # 64 features 36joints

        fd1 += x

        x= x.permute(0,2,1,3) # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)

        x=self.prelus[0](self.txcnns[0](x))
        
        for i in range(1,self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) +x # residual connection
            
        return x
