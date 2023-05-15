from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
from thop.count_hooks import count_convNd
import sys
import os.path as osp
from easydict import EasyDict as edict
from quantize import QConv2d, QuantMeasure, QConvTranspose2d
from basicsr.models.archs.arch_util import (DCNv2Pack, ResidualBlockNoBN,
                                                    make_layer)
from thop import clever_format
ENABLE_TANH = False
from basicsr.models.archs.arch_util import DCNv2Pack as DCN
try:
    from AIM2020_submit.codes.models.archs.deformable_kernels.modules import GlobalDeformKernel2d as GDK
    from AIM2020_submit.codes.models.archs.deformable_kernels.modules import DeformKernelConv2d as DKC
except ImportError:
    raise ImportError('Failed to import Deformable kernels')
C = edict()
"""please config ROOT_dir and user when u first using"""
C.repo_name = 'AGD_VideoSR'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'furnace'))
try:
    from utils.darts_utils import compute_latency_ms_tensorrt as compute_latency
    print("use TensorRT for latency test")
except:
    from utils.darts_utils import compute_latency_ms_pytorch as compute_latency
    print("use PyTorch for latency test")

from slimmable_ops import USConv2d, USBatchNorm2d, USConvTranspose2d, make_divisible

__all__ = ['Temporal_map','Temporal_dot','SpatioTemporal_map','SpatioTemporal_dot','Spatial_dot','Spatial_map','PCD_Align','Align_fea','TSA_Fusion','EPAB','DK_spatial_attention_v2','Conv', 'ConvNorm', 'Conv3x3', 'Conv7x7', 'ConvTranspose2dNorm', 'BasicResidual', 'DwsBlock', 'SkipConnect', 'OPS','OPS_Attention']

latency_lookup_table = {}
table_file_name = "latency_lookup_table_8s.npy"
if osp.isfile(table_file_name):
    latency_lookup_table = np.load(table_file_name).item()

flops_lookup_table = {}
table_file_name = "flops_lookup_table.npy"
if osp.isfile(table_file_name):
    flops_lookup_table = np.load(table_file_name, allow_pickle=True).item()


Conv2d = QConv2d
ConvTranspose2d = QConvTranspose2d
BatchNorm2d = nn.BatchNorm2d    

ENABLE_BN = False

def count_custom(m, x, y):
    m.total_ops += 0

custom_ops={QConv2d: count_convNd, QConvTranspose2d:count_convNd, QuantMeasure: count_custom}
class Temporal_map(nn.Module):
    """ Temporal attention 1st order
    """

    def __init__(self, num_frames,num_channels_in,size,activation,num_feat=10,deformable_groups=8):
        super(Temporal_map, self).__init__()
        self.activation=activation
        self.num_feat=num_feat
        self.conv1 = torch.nn.Conv1d(num_channels_in* size * size, num_feat, 1)
        self.G3 = torch.nn.Conv1d(num_channels_in * size * size,num_feat*size*size, 1)
        self.pool1 = torch.nn.AvgPool1d(num_feat)
        self.ffnn1 = torch.nn.Linear(num_frames, num_frames)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input,input1,act_weights):
            B, T, C, H, W = input.shape
            # print(input.shape)
            map_attention_output=0
            input_4_D_reshaped = torch.transpose(input.reshape([B, T, C * H * W]), 1, 2)
            output_relu= self.relu(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
            map_attention_relu= torch.diag_embed(output_relu)
            map_attention_output_relu = torch.matmul(map_attention_relu, self.G3(input_4_D_reshaped).permute(0, 2, 1))
            map_attention_output_relu= map_attention_output_relu.view([B, T, self.num_feat, H, W])
            map_attention_output=map_attention_output+act_weights[0]*map_attention_output_relu
            output_sigmoid= self.sigmoid(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
            map_attention_sigmoid= torch.diag_embed(output_sigmoid)
            map_attention_output_sigmoid = torch.matmul(map_attention_sigmoid, self.G3(input_4_D_reshaped).permute(0, 2, 1))
            map_attention_output_sigmoid= map_attention_output_sigmoid.view([B, T, self.num_feat, H, W])
            map_attention_output=map_attention_output+act_weights[1]*map_attention_output_sigmoid
            output_softmax = self.softmax(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
            map_attention_softmax = torch.diag_embed(output_softmax)
            map_attention_output_softmax = torch.matmul(map_attention_softmax, self.G3(input_4_D_reshaped).permute(0, 2, 1))
            map_attention_output_softmax = map_attention_output_softmax.view([B, T, self.num_feat, H, W])
            map_attention_output = map_attention_output + act_weights[2] * map_attention_output_softmax
            return map_attention_output


class Temporal_dot(nn.Module):
    '''Temporal attention 2nd order'''

    def __init__(self, num_frames,num_channels_in,size,activation,num_feat=10,deformable_groups=8):
        super(Temporal_dot, self).__init__()
        self.activation=activation
        self.num_feat=num_feat
        self.conv1 = torch.nn.Conv1d(num_channels_in* size * size, num_feat, 1)
        self.conv2 = torch.nn.Conv1d(num_channels_in * size * size, num_feat, 1)
        self.G3 = torch.nn.Conv1d(num_channels_in * size * size,num_feat*size*size, 1)
        self.pool1 = torch.nn.AvgPool1d(num_feat)
        #self.ffnn1 = torch.nn.Linear(num_frames, num_frames)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self,input1,input2,act_weights):
        B,T,C,H,W = input1.shape
        input1_4_D_reshaped = torch.transpose(input1.reshape([B, T, C * H * W]), 1, 2)
        input2_4_D_reshaped = torch.transpose(input2.reshape([B, T, C * H * W]), 1, 2)
        dot_attention_output=0
        output_relu= self.relu(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
        dot_attention_output_relu = torch.matmul(output_relu, self.G3(input1_4_D_reshaped).transpose(1,2))
        dot_attention_output_relu=dot_attention_output_relu.view([B,T,self.num_feat,H,W])
        dot_attention_output=dot_attention_output+act_weights[0]*dot_attention_output_relu
        output_sigmoid= self.sigmoid(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
        dot_attention_output_sigmoid = torch.matmul(output_sigmoid, self.G3(input1_4_D_reshaped).transpose(1,2))
        dot_attention_output_sigmoid=dot_attention_output_sigmoid.view([B,T,self.num_feat,H,W])
        dot_attention_output=dot_attention_output+act_weights[1]*dot_attention_output_sigmoid
        output_softmax= self.softmax(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
        dot_attention_output_softmax = torch.matmul(output_softmax, self.G3(input1_4_D_reshaped).transpose(1,2))
        dot_attention_output_softmax=dot_attention_output_softmax.view([B,T,self.num_feat,H,W])
        dot_attention_output=dot_attention_output+act_weights[2]*dot_attention_output_softmax
        return dot_attention_output

class Channel_map(nn.Module):
    """ Channel attention 1st order
    """

    def __init__(self, num_frames,num_channels_in,size,activation,num_feat=10,deformable_groups=8):
        super(Channel_map, self).__init__()
        self.activation=activation
        self.num_feat=num_feat
        self.conv1 = torch.nn.Conv1d(num_frames* size * size, num_feat, 1)
        self.G3 = torch.nn.Conv1d(num_frames * size * size,num_feat*size*size, 1)
        self.pool1 = torch.nn.AvgPool1d(num_feat)
        self.ffnn1 = torch.nn.Linear(num_channels_in, num_channels_in)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input,input1,act_weights):
            B, T, C, H, W = input.shape
            # print(input.shape)
            map_attention_output=0
            input_4_D_reshaped = torch.transpose(input.permute(0,2,1,3,4).reshape([B, C,T * H * W]), 1, 2)
            output_relu= self.relu(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
            map_attention_relu= torch.diag_embed(output_relu)
            map_attention_output_relu = torch.matmul(map_attention_relu, self.G3(input_4_D_reshaped).permute(0, 2, 1))
            map_attention_output_relu= map_attention_output_relu.view([B, C,self.num_feat, H, W]).permute(0,2,1,3,4)
            map_attention_output=map_attention_output+act_weights[0]*map_attention_output_relu
            output_sigmoid= self.sigmoid(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
            map_attention_sigmoid= torch.diag_embed(output_sigmoid)
            map_attention_output_sigmoid = torch.matmul(map_attention_sigmoid, self.G3(input_4_D_reshaped).permute(0, 2, 1))
            map_attention_output_sigmoid= map_attention_output_sigmoid.view([B, C,self.num_feat, H, W]).permute(0,2,1,3,4)
            map_attention_output=map_attention_output+act_weights[1]*map_attention_output_sigmoid
            output_softmax = self.softmax(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
            map_attention_softmax = torch.diag_embed(output_softmax)
            map_attention_output_softmax = torch.matmul(map_attention_softmax, self.G3(input_4_D_reshaped).permute(0, 2, 1))
            map_attention_output_softmax = map_attention_output_softmax.view([B, C,self.num_feat, H, W]).permute(0,2,1,3,4)
            map_attention_output = map_attention_output + act_weights[2] * map_attention_output_softmax
            return map_attention_output


class Channel_dot(nn.Module):
    '''Channel attention 2nd order'''

    def __init__(self, num_frames,num_channels_in,size,activation,num_feat=10,deformable_groups=8):
        super(Channel_dot, self).__init__()
        self.activation=activation
        self.num_feat=num_feat
        self.conv1 = torch.nn.Conv1d(num_frames* size * size, num_feat, 1)
        self.conv2 = torch.nn.Conv1d(num_frames * size * size, num_feat, 1)
        self.G3 = torch.nn.Conv1d(num_frames * size * size,num_feat*size*size, 1)
        self.pool1 = torch.nn.AvgPool1d(num_feat)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self,input1,input2,act_weights):
        B,T,C,H,W = input1.shape
        input1_4_D_reshaped = torch.transpose(input1.permute(0,2,1,3,4).reshape([B, C, T * H * W]), 1, 2)
        input2_4_D_reshaped = torch.transpose(input2.permute(0,2,1,3,4).reshape([B, C, T * H * W]), 1, 2)
        dot_attention_output=0
        output_relu= self.relu(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
        dot_attention_output_relu = torch.matmul(output_relu, self.G3(input1_4_D_reshaped).transpose(1,2))
        dot_attention_output_relu=dot_attention_output_relu.view([B,C,self.num_feat,H,W]).permute(0,2,1,3,4)
        dot_attention_output=dot_attention_output+act_weights[0]*dot_attention_output_relu
        output_sigmoid= self.sigmoid(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
        dot_attention_output_sigmoid = torch.matmul(output_sigmoid, self.G3(input1_4_D_reshaped).transpose(1,2))
        dot_attention_output_sigmoid=dot_attention_output_sigmoid.view([B,C,self.num_feat,H,W]).permute(0,2,1,3,4)
        dot_attention_output=dot_attention_output+act_weights[1]*dot_attention_output_sigmoid
        output_softmax= self.softmax(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
        dot_attention_output_softmax = torch.matmul(output_softmax, self.G3(input1_4_D_reshaped).transpose(1,2))
        dot_attention_output_softmax=dot_attention_output_softmax.view([B,C,self.num_feat,H,W]).permute(0,2,1,3,4)
        dot_attention_output=dot_attention_output+act_weights[2]*dot_attention_output_softmax
        return dot_attention_output

class SpatioTemporal_map(nn.Module):
    """ Spatiotemporal attention 1st order
    """

    def __init__(self, num_frames,num_channels_in,size,activation,num_feat=10,deformable_groups=8):
        super(SpatioTemporal_map, self).__init__()
        self.activation=activation
        self.num_feat=num_feat
        self.conv1 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.G3 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.pool1 = torch.nn.AvgPool1d(num_feat)
        self.ffnn1 = torch.nn.Linear(num_frames*size*size, num_frames*size*size)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self,input,input1,act_weights):
        B,T,C,H,W = input.shape
        input_4_D_reshaped = input.transpose(1,2).reshape([B,C,T*H*W])
        map_attention_output=0
        output_relu= self.relu(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
        map_attention_relu = torch.diag_embed(output_relu)
        map_attention_output_relu = torch.matmul(map_attention_relu, self.G3(input_4_D_reshaped).permute(0, 2, 1))
        map_attention_output_relu= map_attention_output_relu.view([B, T, H, W, self.num_feat]).permute(0, 1, 4, 2, 3)
        map_attention_output=map_attention_output+act_weights[0]*map_attention_output_relu
        output_sigmoid= self.sigmoid(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
        map_attention_sigmoid = torch.diag_embed(output_sigmoid)
        map_attention_output_sigmoid = torch.matmul(map_attention_sigmoid, self.G3(input_4_D_reshaped).permute(0, 2, 1))
        map_attention_output_sigmoid= map_attention_output_sigmoid.view([B, T, H, W, self.num_feat]).permute(0, 1, 4, 2, 3)
        map_attention_output=map_attention_output+act_weights[1]*map_attention_output_sigmoid
        output_softmax= self.softmax(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
        map_attention_softmax = torch.diag_embed(output_softmax)
        map_attention_output_softmax = torch.matmul(map_attention_softmax, self.G3(input_4_D_reshaped).permute(0, 2, 1))
        map_attention_output_softmax= map_attention_output_softmax.view([B, T, H, W, self.num_feat]).permute(0, 1, 4, 2, 3)
        map_attention_output=map_attention_output+act_weights[2]*map_attention_output_softmax
        return map_attention_output

class SpatioTemporal_dot(nn.Module):
    """ SpatioTemporal attention 2nd order
    """

    def __init__(self, num_frames,num_channels_in,size,activation,num_feat=10,deformable_groups=8):
        super(SpatioTemporal_dot, self).__init__()
        self.activation=activation
        self.num_feat=num_feat
        self.conv1 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.conv2 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.G3 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.pool1 = torch.nn.AvgPool1d(num_feat)
        #self.ffnn1 = torch.nn.Linear(num_frames*size*size, num_frames*size*size)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self,input1,input2,act_weights):
        B,T,C,H,W = input1.shape
        dot_attention_output=0
        input1_4_D_reshaped = input1.transpose(1,2).reshape([B,C,T * H * W])
        input2_4_D_reshaped = input2.transpose(1, 2).reshape([B, C, T * H * W])
        output_relu = self.relu(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
        dot_attention_output_relu= torch.matmul(output_relu, self.G3(input1_4_D_reshaped).permute(0, 2, 1))
        dot_attention_output_relu= dot_attention_output_relu.view([B, T, H, W, self.num_feat]).permute(0, 1, 4, 2, 3)
        dot_attention_output=dot_attention_output+act_weights[0]*dot_attention_output_relu
        output_sigmoid= self.sigmoid(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
        dot_attention_output_sigmoid= torch.matmul(output_sigmoid, self.G3(input1_4_D_reshaped).permute(0, 2, 1))
        dot_attention_output_sigmoid= dot_attention_output_sigmoid.view([B, T, H, W, self.num_feat]).permute(0, 1, 4, 2, 3)
        dot_attention_output=dot_attention_output+act_weights[1]*dot_attention_output_sigmoid
        output_softmax= self.softmax(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
        dot_attention_output_softmax= torch.matmul(output_softmax, self.G3(input1_4_D_reshaped).permute(0, 2, 1))
        dot_attention_output_softmax= dot_attention_output_softmax.view([B, T, H, W, self.num_feat]).permute(0, 1, 4, 2, 3)
        dot_attention_output=dot_attention_output+act_weights[0]*dot_attention_output_softmax
        return dot_attention_output

class Spatial_map(nn.Module):
    """ Spatial attention (1st order)
    """

    def __init__(self, num_frames,num_channels_in,size,activation,num_feat=10,deformable_groups=8):
        super(Spatial_map, self).__init__()
        self.activation=activation
        self.num_feat=num_feat
        self.conv1 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.G3 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.pool1 = torch.nn.AvgPool1d(num_feat)
        self.ffnn1 = torch.nn.Linear(size*size, size*size)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self,input,input1,act_weights):
        B,T,C,H,W = input.shape
        input_4_D_reshaped = input.transpose(1,2).reshape([B,C,T,H*W])
        dp_all_relu=[]
        for i in range(input_4_D_reshaped.shape[2]):
            inp=input_4_D_reshaped[:,:,i,:]
            output_relu= self.relu(self.ffnn1(self.pool1(self.conv1(inp).transpose(1, 2)).squeeze(-1)))
            map_attention_relu= torch.diag_embed(output_relu)
            map_attention_output_relu= torch.matmul(map_attention_relu, self.G3(inp).transpose(1,2))
            map_attention_output_relu=map_attention_output_relu.view([B,H,W,self.num_feat]).permute(0,3,1,2)
            dp_all_relu.append(map_attention_output_relu)
        dp_all_relu= torch.stack(dp_all_relu,dim=1)
        dp_all_sigmoid=[]
        for i in range(input_4_D_reshaped.shape[2]):
            inp=input_4_D_reshaped[:,:,i,:]
            output_sigmoid= self.sigmoid(self.ffnn1(self.pool1(self.conv1(inp).transpose(1, 2)).squeeze(-1)))
            map_attention_sigmoid= torch.diag_embed(output_sigmoid)
            map_attention_output_sigmoid= torch.matmul(map_attention_sigmoid, self.G3(inp).transpose(1,2))
            map_attention_output_sigmoid=map_attention_output_sigmoid.view([B,H,W,self.num_feat]).permute(0,3,1,2)
            dp_all_sigmoid.append(map_attention_output_sigmoid)
        dp_all_sigmoid= torch.stack(dp_all_sigmoid,dim=1)
        dp_all_softmax=[]
        for i in range(input_4_D_reshaped.shape[2]):
            inp=input_4_D_reshaped[:,:,i,:]
            output_softmax= self.softmax(self.ffnn1(self.pool1(self.conv1(inp).transpose(1, 2)).squeeze(-1)))
            map_attention_softmax= torch.diag_embed(output_softmax)
            map_attention_output_softmax= torch.matmul(map_attention_softmax, self.G3(inp).transpose(1,2))
            map_attention_output_softmax=map_attention_output_softmax.view([B,H,W,self.num_feat]).permute(0,3,1,2)
            dp_all_softmax.append(map_attention_output_softmax)
        dp_all_softmax= torch.stack(dp_all_softmax,dim=1)
        dp_all=dp_all_relu*act_weights[0]+dp_all_sigmoid*act_weights[1]+dp_all_softmax*act_weights[2]
        return dp_all

class Spatial_dot(nn.Module):
    """ Spatial attention (2nd order)
    """

    def __init__(self, num_frames,num_channels_in,size,activation,num_feat=10,deformable_groups=8):
        super(Spatial_dot, self).__init__()
        self.activation=activation
        self.num_feat=num_feat
        self.conv1 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.conv2 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.G3 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.pool1 = torch.nn.AvgPool1d(num_feat)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self,input1,input2,act_weights):
        B,T,C,H,W = input1.shape
        input1_4_D_reshaped = input1.transpose(1,2).reshape([B,C,T,H*W])
        input2_4_D_reshaped = input2.transpose(1, 2).reshape([B, C,T,H*W])
        dp_all_relu=[]
        for i in range(input1_4_D_reshaped.shape[2]):
            inp1=input1_4_D_reshaped[:,:,i,:]
            inp2=input2_4_D_reshaped[:,:,i,:]
            output_relu= self.relu(torch.matmul(self.conv1(inp1).transpose(1,2),self.conv2(inp2)))
            dot_attention_output_relu = torch.matmul(output_relu, self.G3(inp1).transpose(1,2))
            dot_attention_output_relu=dot_attention_output_relu.view([B,H,W,self.num_feat]).permute(0,3,1,2)
            dp_all_relu.append(dot_attention_output_relu)
        dp_all_relu=torch.stack(dp_all_relu,dim=1)
        dp_all_sigmoid=[]
        for i in range(input1_4_D_reshaped.shape[2]):
            inp1=input1_4_D_reshaped[:,:,i,:]
            inp2=input2_4_D_reshaped[:,:,i,:]
            output_sigmoid= self.sigmoid(torch.matmul(self.conv1(inp1).transpose(1,2),self.conv2(inp2)))
            dot_attention_output_sigmoid= torch.matmul(output_sigmoid, self.G3(inp1).transpose(1,2))
            dot_attention_output_sigmoid=dot_attention_output_sigmoid.view([B,H,W,self.num_feat]).permute(0,3,1,2)
            dp_all_sigmoid.append(dot_attention_output_sigmoid)
        dp_all_sigmoid = torch.stack(dp_all_sigmoid, dim=1)
        dp_all_softmax=[]
        for i in range(input1_4_D_reshaped.shape[2]):
            inp1=input1_4_D_reshaped[:,:,i,:]
            inp2=input2_4_D_reshaped[:,:,i,:]
            output_softmax= self.softmax(torch.matmul(self.conv1(inp1).transpose(1,2),self.conv2(inp2)))
            dot_attention_output_softmax = torch.matmul(output_softmax, self.G3(inp1).transpose(1,2))
            dot_attention_output_softmax=dot_attention_output_softmax.view([B,H,W,self.num_feat]).permute(0,3,1,2)
            dp_all_softmax.append(dot_attention_output_softmax)
        dp_all_softmax=torch.stack(dp_all_softmax,dim=1)
        dp_all=act_weights[0]*dp_all_relu+act_weights[1]*dp_all_sigmoid+act_weights[2]*dp_all_softmax
        return dp_all

class PCD_Align(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    Reduce PCD cascade to 2 levels
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCD_Align, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(2, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = Conv(num_feat * 2, num_feat, 3, 1,
                                                 1)
            if i == 2:
                self.offset_conv2[level] = Conv(num_feat, num_feat, 3, 1,
                                                     1)
            else:
                self.offset_conv2[level] = Conv(num_feat * 2, num_feat, 3,
                                                     1, 1)
                self.offset_conv3[level] = Conv(num_feat, num_feat, 3, 1,
                                                     1)
            self.dcn_pack[level] = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                padding=1,
                deformable_groups=deformable_groups)

            if i < 2:
                self.feat_conv[level] = Conv(num_feat * 2, num_feat, 3, 1,
                                                  1)

        # Cascading dcn
        self.cas_offset_conv1 = Conv(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = Conv(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(
            num_feat,
            num_feat,
            3,
            padding=1,
            deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(2, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 2:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 2:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat

    def forward_flops(self,size1,size2,name):

        # for i in range(self.num_frames):
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            nbr_fea_l = [torch.randn(size1).cuda(), torch.randn(size2).cuda()]
            flops, _ = profile(PCD_Align().cuda(), inputs=(nbr_fea_l, nbr_fea_l))
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops

class Align_fea(nn.Module):
    def __init__(self, nf=64,  groups=8):
        super(Align_fea, self).__init__()
        self.offset_conv1 = Conv(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.offset_conv2 = DKC((4, 4), nf, nf, 3, 1, 1, groups=nf)
        self.offset_conv3 = DKC((4, 4), nf, nf, 3, 1, 1, groups=nf)
        self.offset_conv4 = DKC((4, 4), nf, nf, 3, 1, 1, groups=nf)
        self.offset_conv5 = DKC((4, 4), nf, nf, 3, 1, 1, groups=nf)
        self.offset_conv6 = DKC((4, 4), nf, nf, 3, 1, 1, groups=nf)
        self.offset_conv7 = DKC((4, 4), nf, nf, 3, 1, 1, groups=nf)
        self.dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):

        offset = torch.cat([nbr_fea_l, ref_fea_l], dim=1)
        offset_org = self.lrelu(self.offset_conv1(offset))

        offset_GDK = self.lrelu(self.offset_conv2(offset_org))
        offset_DKC = self.lrelu(self.offset_conv3(offset_GDK))
        offset_DKC = self.lrelu(self.offset_conv4(offset_DKC))
        offset_DKC = self.lrelu(self.offset_conv5(offset_DKC))
        offset_DKC = self.lrelu(self.offset_conv6(offset_DKC))
        offset_DKC = self.lrelu(self.offset_conv7(offset_DKC))
        offset = offset_DKC

        fea = self.lrelu(self.dcnpack(nbr_fea_l, offset))

        return fea

    def forward_flops(self,size1,name):

        # for i in range(self.num_frames):
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            nbr_fea_l=torch.randn(size1).cuda()
            flops, _ = profile(Align_fea().cuda(), inputs=(nbr_fea_l, nbr_fea_l))
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops

class SimpleNonLocal_Block_Video(nn.Module):
    def __init__(self, nf, mode):
        super(SimpleNonLocal_Block_Video, self).__init__()

        self.convx1 = nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.convx2 = nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.convx4 = nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.mode = mode

        assert mode in ['spatial', 'channel', 'temporal'], 'Mode from NL Block not recognized.'

    def forward(self, x1):
        if self.mode == 'channel':
            x = x1.clone()
            xA = torch.sigmoid(self.convx1(x))
            xB = self.convx2(x)*xA
            x = self.convx4(xB)

        elif self.mode == 'temporal':
            x = x1.permute(0, 2, 1, 3, 4).contiguous()  # BTCHW to BCTHW
            xA = torch.sigmoid(self.convx1(x))
            xB = self.convx2(x)*xA
            xB = self.convx4(xB)
            x = xB.permute(0, 2, 1, 3, 4).contiguous()  # BCTHW to BTCHW

        return x + x1
class SimpleNonLocal_Block_Video_NAS(nn.Module):
    def __init__(self, nf, mode):
        super(SimpleNonLocal_Block_Video_NAS, self).__init__()

        self.convx1 = nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.convx2 = nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.convx4 = nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.mode = mode
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        assert mode in ['spatial', 'channel', 'temporal'], 'Mode from NL Block not recognized.'

    def forward(self, x1,act_weights):
        if self.mode == 'channel':
            x = x1.clone()
            xA = torch.sigmoid(self.convx1(x))
            xB = self.convx2(x)*xA
            x = self.convx4(xB)

        elif self.mode == 'temporal':
            x = x1.permute(0, 2, 1, 3, 4).contiguous()  # BTCHW to BCTHW
            intm=self.convx1(x)
            xA = self.relu(intm)*act_weights[0]+self.sigmoid(intm)*act_weights[1]+self.softmax(intm)*act_weights[2]
            xB = self.convx2(x)*xA
            xB = self.convx4(xB)
            x = xB.permute(0, 2, 1, 3, 4).contiguous()  # BCTHW to BTCHW

        return x + x1

class EPAB_SpatioChannel(nn.Module):
    def __init__(self, nf=128, num_frames=7):
        super(EPAB_SpatioChannel, self).__init__()
        self.NL_Block_Vid_channel = SimpleNonLocal_Block_Video_NAS(num_frames, 'channel')
        #self.fusion_conv = Conv(nf * num_frames, nf, 3, 1, 1, bias=True)


    def forward(self, F,act_weights):
        B, T, C, H, W = F.shape
        channel = self.NL_Block_Vid_channel(F,act_weights)
        out = channel +  F
        #out = self.fusion_conv(out.view(B, -1, H, W))
        return out

    def forward_flops(self,size1,name):

        # for i in range(self.num_frames):
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            inp = torch.randn(size1).cuda()
            flops, _ = profile(EPAB_SpatioChannel(64,7).cuda(), inputs=(inp.cuda(),))
            #flops, _ = profile(self.align, inputs=(nbr_fea_l, nbr_fea_l))
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops

class EPAB_SpatioTemporal(nn.Module):
    def __init__(self, nf=128, num_frames=7):
        super(EPAB_SpatioTemporal, self).__init__()
        self.NL_Block_Vid_temporal = SimpleNonLocal_Block_Video_NAS(nf, 'temporal')
        #self.fusion_conv = Conv(nf * num_frames, nf, 3, 1, 1, bias=True)
    def forward(self, F,act_weights):
        B,T,C,H,W=F.shape
        temporal = self.NL_Block_Vid_temporal(F,act_weights)
        out = temporal +  F
        #out = self.fusion_conv(out.view(B, -1, H, W))
        return out

    def forward_flops(self,size1,name):

        # for i in range(self.num_frames):
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            inp = torch.randn(size1).cuda()
            flops, _ = profile(EPAB_SpatioTemporal(64,7).cuda(), inputs=(inp.cuda(),))
            #flops, _ = profile(self.align, inputs=(nbr_fea_l, nbr_fea_l))
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops

class EPAB(nn.Module):
    def __init__(self, nf=128, num_frames=7):
        super(EPAB, self).__init__()
        self.NL_Block_Vid_channel = SimpleNonLocal_Block_Video(num_frames, 'channel')
        self.NL_Block_Vid_temporal = SimpleNonLocal_Block_Video(nf, 'temporal')

    def forward(self, F):

        channel = self.NL_Block_Vid_channel(F)
        temporal = self.NL_Block_Vid_temporal(F)

        out = channel + temporal + F

        return out

    def forward_flops(self,size1,name):

        # for i in range(self.num_frames):
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            inp = torch.randn(size1).cuda()
            flops, _ = profile(EPAB(64,7).cuda(), inputs=(inp.cuda(),))
            #flops, _ = profile(self.align, inputs=(nbr_fea_l, nbr_fea_l))
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops

class DK_spatial_attention_v2(nn.Module):
    '''Deformable kernel spatial attention module
    for last 3 x2 levels'''

    def __init__(self, nf=64):
        super(DK_spatial_attention_v2, self).__init__()
        self.fusion_conv = Conv(self.nf * self.num_frames, self.nf, 3, 1, 1, bias=True)
        self.DKC = nn.Sequential(Conv(nf, nf, kernel_size=3, stride=2, padding=1, bias=True),
                                 nn.ReLU(inplace=True),
                                 DKC((4, 4), nf, nf, kernel_size=3, stride=1, padding=1, groups=nf),
                                 nn.ReLU(inplace=True),
                                 DKC((4, 4), nf, nf, kernel_size=3, stride=1, padding=1, groups=nf),
                                 nn.ReLU(inplace=True),
                                 DKC((4, 4), nf, nf, kernel_size=3, stride=1, padding=1, groups=nf),
                                 nn.ReLU(inplace=True),
                                 DKC((4, 4), nf, nf, kernel_size=3, stride=1, padding=1, groups=nf),
                                 nn.ReLU(inplace=True),
                                 DKC((4, 4), nf, nf, kernel_size=3, stride=1, padding=1, groups=nf),
                                 nn.ReLU(inplace=True),
                                 Conv(nf, 4 * nf, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.PixelShuffle(2),
                                 Conv(nf, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        B,T,C,H,W=x.shape
        x= self.fusion_conv(x.view(B, -1, H, W))
        return x * self.DKC(x)

    def forward_flops(self,size1,name):

        # for i in range(self.num_frames):
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            inp = torch.randn(size1).cuda()
            flops, _ = profile(DK_spatial_attention_v2().cuda(), inputs=(inp.cuda(),))
            #flops, _ = profile(self.align, inputs=(nbr_fea_l, nbr_fea_l))
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops

class TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, nf=64, nframes=5, center=2):
        super(TSA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = Conv(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = Conv(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = Conv(nframes * nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = Conv(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = Conv(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = Conv(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = Conv(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = Conv(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = Conv(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = Conv(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = Conv(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = Conv(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = Conv(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea

    def forward_flops(self,size1,name):

        # for i in range(self.num_frames):
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            inp = torch.randn(size1).cuda()
            flops, _ = profile( TSA_Fusion(nf=64, nframes=7, center=3).cuda(), inputs=(inp.cuda(),))
            #flops, _ = profile(self.align, inputs=(nbr_fea_l, nbr_fea_l))
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops

class Conv(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, slimmable=False, width_mult_list=[1.]):
        super(Conv, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)

        if slimmable:
            self.conv = USConv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias, width_mult_list=width_mult_list)

        else:
            self.conv = Conv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias)
            
    
    def set_ratio(self, ratio):
        assert self.slimmable
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv.set_ratio(ratio)

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = Conv(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = Conv(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "Conv_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = Conv._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "Conv_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = Conv._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     flops /= 4

        return flops, (c_out, h_out, w_out)


    def forward(self, x, quantize=False):
        x = self.conv(x, quantize=quantize)
        return x


class ConvNorm(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, slimmable=False, width_mult_list=[1.]):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)

        if slimmable:
            self.conv = USConv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias, width_mult_list=width_mult_list)
            self.bn = USBatchNorm2d(C_out, width_mult_list)
            self.relu = nn.ReLU(inplace=True)
            
        else:
            self.conv = Conv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias)
            self.bn = BatchNorm2d(C_out)
            self.relu = nn.ReLU(inplace=True)
            
    
    def set_ratio(self, ratio):
        assert self.slimmable
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv.set_ratio(ratio)
        self.bn.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = ConvNorm._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvNorm._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     flops /= 4

        return flops, (c_out, h_out, w_out)


    def forward(self, x, quantize=False):
        x = self.conv(x, quantize=quantize)
        if ENABLE_BN:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTranspose2dNorm(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=2, padding=None, dilation=1, groups=1, bias=False, slimmable=True, width_mult_list=[1.]):
        super(ConvTranspose2dNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        self.padding = 1
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)

        if slimmable:
            self.conv = USConvTranspose2d(C_in, C_out, kernel_size, stride, padding=self.padding,  output_padding=1, dilation=dilation, groups=self.groups, bias=bias, width_mult_list=width_mult_list)
            self.bn = USBatchNorm2d(C_out, width_mult_list)
            self.relu = nn.ReLU(inplace=True)
            
        else:
            self.conv = ConvTranspose2d(C_in, C_out, kernel_size, stride, padding=self.padding,  output_padding=1, dilation=dilation, groups=self.groups, bias=bias)
            self.bn = BatchNorm2d(C_out)
            self.relu = nn.ReLU(inplace=True)
            
    
    def set_ratio(self, ratio):
        assert self.slimmable
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv.set_ratio(ratio)
        self.bn.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvTranspose2dNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvTranspose2dNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "ConvTranspose2dNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = ConvTranspose2dNorm._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "ConvTranspose2dNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvTranspose2dNorm._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     flops /= 4

        return flops, (c_out, h_out, w_out)


    def forward(self, x, quantize=False):
        x = self.conv(x, quantize=quantize)
        if ENABLE_BN:
            x = self.bn(x)
        x = self.relu(x)
        return x


class Conv7x7(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=7, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(Conv7x7, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        if slimmable:
            self.conv1 = USConv2d(C_in, C_out, 7, stride, padding=3, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
        else:
            self.conv1 = Conv2d(C_in, C_out, 7, stride, padding=3, dilation=dilation, groups=groups, bias=False)
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=7, stride=1, dilation=1, groups=1):
        layer = Conv7x7(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=7, stride=1, dilation=1, groups=1):
        layer = Conv7x7(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, int(self.C_in * self.ratio[0]) %d"%(c_in, int(self.C_in * self.ratio[0]))
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "Conv7x7_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = Conv7x7._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "Conv7x7_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = Conv7x7._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     flops /= 4

        return flops, (c_out, h_out, w_out)


    def forward(self, x, quantize=False):
        identity = x
        out = self.conv1(x, quantize=quantize)

        return out


class Conv3x3(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(Conv3x3, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        if slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn1 = nn.BatchNorm2d(C_out)
            self.bn1 = BatchNorm2d(C_out)
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = Conv3x3(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = Conv3x3(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, int(self.C_in * self.ratio[0]) %d"%(c_in, int(self.C_in * self.ratio[0]))
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "Conv3x3_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = Conv3x3._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "Conv3x3_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = Conv3x3._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     flops /= 4

        return flops, (c_out, h_out, w_out)


    def forward(self, x, quantize=False):
        identity = x
        out = self.conv1(x, quantize=quantize)
        if ENABLE_BN:
            out = self.bn1(out)
        out = self.relu(out)
        return out


class BasicResidual(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(BasicResidual, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)

        if self.slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
            self.conv2 = USConv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn2 = USBatchNorm2d(C_out, width_mult_list)

            self.skip = USConv2d(C_in, C_out, 1, stride, padding=0, dilation=dilation, groups=1, bias=False, width_mult_list=width_mult_list)
            self.bn3 = USBatchNorm2d(C_out, width_mult_list)
       
        else:
            self.conv1 = Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn1 = nn.BatchNorm2d(C_out)
            self.bn1 = BatchNorm2d(C_out)
            self.conv2 = Conv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn2 = nn.BatchNorm2d(C_out)
            self.bn2 = BatchNorm2d(C_out)

            if self.C_in != self.C_out or self.stride != 1:
                self.skip = Conv2d(C_in, C_out, 1, stride, padding=0, dilation=dilation, groups=1, bias=False)
                self.bn3 = BatchNorm2d(C_out)
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])
        self.conv2.set_ratio((ratio[1], ratio[1]))
        self.bn2.set_ratio(ratio[1])

        if hasattr(self, 'skip'):
            self.skip.set_ratio(ratio)
            self.bn3.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in%d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = BasicResidual._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = BasicResidual._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     flops /= 4

        return flops, (c_out, h_out, w_out)


    def forward(self, x, quantize=False):
        identity = x
        out = self.conv1(x, quantize=quantize)
        if ENABLE_BN:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out, quantize=quantize)
        if ENABLE_BN:
            out = self.bn2(out)

        if hasattr(self, 'skip'):
            identity = self.skip(identity, quantize=quantize)
            if ENABLE_BN:
                identity = self.bn3(identity)
            
        out += identity
        out = self.relu(out)

        return out


class SkipConnect(nn.Module):
    def __init__(self, C_in, C_out, stride=1, slimmable=True, width_mult_list=[1.]):
        super(SkipConnect, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0, 'C_out=%d'%C_out
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)

        self.kernel_size = 1
        self.padding = 0

        if slimmable:
            self.conv = USConv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False, width_mult_list=width_mult_list)
            self.bn = USBatchNorm2d(C_out, width_mult_list)
            self.relu = nn.ReLU(inplace=True)

        # elif stride == 2 or C_in != C_out:
        else:
            self.conv = Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
            self.bn = BatchNorm2d(C_out)
            self.relu = nn.ReLU(inplace=True)


    def set_ratio(self, ratio):
        assert len(ratio) == 2

        self.ratio = ratio
        self.conv.set_ratio(ratio)
        self.bn.set_ratio(ratio[1])


    @staticmethod
    def _flops(h, w, C_in, C_out, stride=1):
        layer = SkipConnect(C_in, C_out, stride, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, stride=1):
        layer = SkipConnect(C_in, C_out, stride, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "SkipConnect_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = SkipConnect._latency(h_in, w_in, c_in, c_out, self.stride)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "SkipConnect_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = SkipConnect._flops(h_in, w_in, c_in, c_out, self.stride)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     flops /= 4

        return flops, (c_out, h_out, w_out)


    def forward(self, x, quantize=False):
        if hasattr(self, 'conv'):
            out = self.conv(x, quantize=quantize)
            if ENABLE_BN:
                out = self.bn(out)
            out = self.relu(out)
        else:
            out = x

        return out


class DwsBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(DwsBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)

        if self.slimmable:
            self.conv1 = USConv2d(C_in, C_in*4, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_in*4, width_mult_list)

            self.conv2 = USConv2d(C_in*4, C_in*4, 3, stride, padding=dilation, dilation=dilation, groups=C_in*4, bias=False, width_mult_list=width_mult_list)
            self.bn2 = USBatchNorm2d(C_in*4, width_mult_list)

            self.conv3 = USConv2d(C_in*4, C_out, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn3 = USBatchNorm2d(C_out, width_mult_list)

            self.skip = USConv2d(C_in, C_out, 1, stride, padding=0, dilation=dilation, groups=1, bias=False, width_mult_list=width_mult_list)
            self.bn4 = USBatchNorm2d(C_out, width_mult_list)
       
        else:
            self.conv1 = Conv2d(C_in, C_in*4, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False)
            self.bn1 = BatchNorm2d(C_in*4)

            self.conv2 = Conv2d(C_in*4, C_in*4, 3, stride, padding=dilation, dilation=dilation, groups=C_in*4, bias=False)
            self.bn2 = BatchNorm2d(C_in*4)

            self.conv3 = Conv2d(C_in*4, C_out, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False)
            self.bn3 = BatchNorm2d(C_out)

            if self.C_in != self.C_out or self.stride != 1:
                self.skip = Conv2d(C_in, C_out, 1, stride, padding=0, dilation=dilation, groups=1, bias=False)
                self.bn4 = BatchNorm2d(C_out)

    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio((ratio[0], 1))
        self.bn1.set_ratio(1)
        self.conv2.set_ratio((1, 1))
        self.bn2.set_ratio(1)
        self.conv3.set_ratio((1, ratio[1]))
        self.bn3.set_ratio(ratio[1])

        if hasattr(self, 'skip'):
            self.skip.set_ratio(ratio)
            self.bn4.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = DwsBlock(C_in, C_out, kernel_size, stride, dilation, groups=1, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = DwsBlock(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "DwsBlock_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = DwsBlock._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "DwsBlock_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = DwsBlock._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     ratio_dws = 3*3 / (3*3 + self.C_out)
        #     flops = ratio_dws * flops + (1-ratio_dws) * flops / 4

        return flops, (c_out, h_out, w_out)


    def forward(self, x, quantize=False):
        identity = x

        out = self.conv1(x, quantize=quantize)
        if ENABLE_BN:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out, quantize=False)
        if ENABLE_BN:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out, quantize=quantize)
        if ENABLE_BN:
            out = self.bn3(out)

        if hasattr(self, 'skip'):
            identity = self.skip(identity, quantize=quantize)
            if ENABLE_BN:
                identity = self.bn4(identity)
            
        out += identity
        out = self.relu(out)

        return out


OPS = {
    'skip' : lambda C_in, C_out, stride, slimmable, width_mult_list: SkipConnect(C_in, C_out, stride, slimmable, width_mult_list),
    'conv3x3' : lambda C_in, C_out, stride, slimmable, width_mult_list: Conv3x3(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
    'conv3x3_d2' : lambda C_in, C_out, stride, slimmable, width_mult_list: Conv3x3(C_in, C_out, kernel_size=3, stride=stride, dilation=2, slimmable=slimmable, width_mult_list=width_mult_list),
    'conv3x3_d4' : lambda C_in, C_out, stride, slimmable, width_mult_list: Conv3x3(C_in, C_out, kernel_size=3, stride=stride, dilation=4, slimmable=slimmable, width_mult_list=width_mult_list),
    'residual' : lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
    'dwsblock' : lambda C_in, C_out, stride, slimmable, width_mult_list: DwsBlock(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
}

'''OPS_Attention ={
    'temporal_map': lambda num_frames, num_channels_in, size,nf: Temporal_map(num_frames,num_channels_in,size,"relu",num_feat=16),
    'temporal_dot':  lambda num_frames, num_channels_in, size,nf: Temporal_dot(num_frames,num_channels_in,size,"relu",num_feat=16),
    'spatial_map' :  lambda num_frames, num_channels_in, size,nf: Spatial_map(num_frames,num_channels_in,size,"relu",num_feat=16),
    'spatial_dot' :  lambda num_frames, num_channels_in, size,nf: Spatial_dot(num_frames,num_channels_in,size,"relu",num_feat=16),
    'spatiotemporal_map': lambda num_frames, num_channels_in, size,nf: SpatioTemporal_map(num_frames,num_channels_in,size,"relu",num_feat=16),
    'spatiotemporal_dot': lambda num_frames, num_channels_in, size,nf: SpatioTemporal_dot(num_frames,num_channels_in,size,"relu",num_feat=16),
    'channel_map':  lambda num_frames,num_channels_in,size,nf: Channel_map(num_frames,num_channels_in,size,"relu",num_feat=7),
    'channel_dot':  lambda num_frames,num_channels_in,size,nf: Channel_dot(num_frames,num_channels_in,size,"relu",num_feat=7),
}'''

OPS_Attention={
    'epab_spatiochannel': lambda nf, num_frames: EPAB_SpatioChannel(nf,num_frames),
    'epab_spatiotemporal': lambda nf, num_frames: EPAB_SpatioTemporal(nf,num_frames),
}
