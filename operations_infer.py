import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import PRIMITIVES, PRIMITIVES_attention



class Temporal_map(nn.Module):
    """ Temporal attention 1st order
    """

    def __init__(self, num_frames,num_channels_in,size,activation,act,num_feat=10,deformable_groups=8):
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
        self.act=act


    def forward(self, input,input1):
            B, T, C, H, W = input.shape
            # print(input.shape)
            act=self.act
            print(act)
            input_4_D_reshaped = torch.transpose(input.reshape([B, T, C * H * W]), 1, 2)
            if act==0:
                output_relu= self.relu(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
                map_attention_relu= torch.diag_embed(output_relu)
                map_attention_output_relu = torch.matmul(map_attention_relu, self.G3(input_4_D_reshaped).permute(0, 2, 1))
                map_attention_output_relu= map_attention_output_relu.view([B, T, self.num_feat, H, W])
                map_attention_output=map_attention_output_relu
            elif act==1:
                output_sigmoid= self.sigmoid(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
                map_attention_sigmoid= torch.diag_embed(output_sigmoid)
                map_attention_output_sigmoid = torch.matmul(map_attention_sigmoid, self.G3(input_4_D_reshaped).permute(0, 2, 1))
                map_attention_output_sigmoid= map_attention_output_sigmoid.view([B, T, self.num_feat, H, W])
                map_attention_output=map_attention_output_sigmoid
            elif act==2:
                output_softmax = self.softmax(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
                map_attention_softmax = torch.diag_embed(output_softmax)
                map_attention_output_softmax = torch.matmul(map_attention_softmax, self.G3(input_4_D_reshaped).permute(0, 2, 1))
                map_attention_output_softmax = map_attention_output_softmax.view([B, T, self.num_feat, H, W])
                map_attention_output = map_attention_output_softmax
            return map_attention_output


class Temporal_dot(nn.Module):
    '''Temporal attention 2nd order'''

    def __init__(self, num_frames,num_channels_in,size,activation,act,num_feat=10,deformable_groups=8):
        super(Temporal_dot, self).__init__()
        self.activation=activation
        self.num_feat=num_feat
        self.conv1 = torch.nn.Conv1d(num_channels_in* size * size, num_feat, 1)
        self.conv2 = torch.nn.Conv1d(num_channels_in * size * size, num_feat, 1)
        self.G3 = torch.nn.Conv1d(num_channels_in * size * size,num_feat*size*size, 1)
        self.pool1 = torch.nn.AvgPool1d(num_feat)
        self.ffnn1 = torch.nn.Linear(num_frames, num_frames)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.act=act


    def forward(self,input1,input2):
        B,T,C,H,W = input1.shape
        input1_4_D_reshaped = torch.transpose(input1.reshape([B, T, C * H * W]), 1, 2)
        input2_4_D_reshaped = torch.transpose(input2.reshape([B, T, C * H * W]), 1, 2)
        act=self.act
        print(act)
        if act==0:
         output_relu= self.relu(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
         dot_attention_output_relu = torch.matmul(output_relu, self.G3(input1_4_D_reshaped).transpose(1,2))
         dot_attention_output_relu=dot_attention_output_relu.view([B,T,self.num_feat,H,W])
         dot_attention_output=dot_attention_output_relu
        elif act==1:
         output_sigmoid= self.sigmoid(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
         dot_attention_output_sigmoid = torch.matmul(output_sigmoid, self.G3(input1_4_D_reshaped).transpose(1,2))
         dot_attention_output_sigmoid=dot_attention_output_sigmoid.view([B,T,self.num_feat,H,W])
         dot_attention_output=dot_attention_output_sigmoid
        elif act==2:
         output_softmax= self.softmax(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
         dot_attention_output_softmax = torch.matmul(output_softmax, self.G3(input1_4_D_reshaped).transpose(1,2))
         dot_attention_output_softmax=dot_attention_output_softmax.view([B,T,self.num_feat,H,W])
         dot_attention_output=dot_attention_output_softmax
        return dot_attention_output

class Channel_map(nn.Module):
    """ Channel attention 1st order
    """

    def __init__(self, num_frames,num_channels_in,size,activation,act,num_feat=10,deformable_groups=8):
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


    def forward(self, input,input1):
            B, T, C, H, W = input.shape
            act=self.act
            print(act)
            input_4_D_reshaped = torch.transpose(input.permute(0,2,1,3,4).reshape([B, C,T * H * W]), 1, 2)
            if act==0:
              output_relu= self.relu(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
              map_attention_relu= torch.diag_embed(output_relu)
              map_attention_output_relu = torch.matmul(map_attention_relu, self.G3(input_4_D_reshaped).permute(0, 2, 1))
              map_attention_output_relu= map_attention_output_relu.view([B, C,self.num_feat, H, W]).permute(0,2,1,3,4)
              map_attention_output=map_attention_output_relu
            elif act==1:
              output_sigmoid= self.sigmoid(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
              map_attention_sigmoid= torch.diag_embed(output_sigmoid)
              map_attention_output_sigmoid = torch.matmul(map_attention_sigmoid, self.G3(input_4_D_reshaped).permute(0, 2, 1))
              map_attention_output_sigmoid= map_attention_output_sigmoid.view([B, C,self.num_feat, H, W]).permute(0,2,1,3,4)
              map_attention_output=map_attention_output_sigmoid
            elif act==2:
             output_softmax = self.softmax(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
             map_attention_softmax = torch.diag_embed(output_softmax)
             map_attention_output_softmax = torch.matmul(map_attention_softmax, self.G3(input_4_D_reshaped).permute(0, 2, 1))
             map_attention_output_softmax = map_attention_output_softmax.view([B, C,self.num_feat, H, W]).permute(0,2,1,3,4)
             map_attention_output = map_attention_output_softmax
            return map_attention_output


class Channel_dot(nn.Module):
    '''Channel attention 2nd order'''

    def __init__(self, num_frames,num_channels_in,size,activation,act,num_feat=10,deformable_groups=8):
        super(Channel_dot, self).__init__()
        self.activation=activation
        self.num_feat=num_feat
        self.act=act
        self.conv1 = torch.nn.Conv1d(num_frames* size * size, num_feat, 1)
        self.conv2 = torch.nn.Conv1d(num_frames * size * size, num_feat, 1)
        self.G3 = torch.nn.Conv1d(num_frames * size * size,num_feat*size*size, 1)
        self.pool1 = torch.nn.AvgPool1d(num_feat)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self,input1,input2):
        B,T,C,H,W = input1.shape
        input1_4_D_reshaped = torch.transpose(input1.permute(0,2,1,3,4).reshape([B, C, T * H * W]), 1, 2)
        input2_4_D_reshaped = torch.transpose(input2.permute(0,2,1,3,4).reshape([B, C, T * H * W]), 1, 2)
        act=self.act
        print(act)
        if act==0:
          output_relu= self.relu(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
          dot_attention_output_relu = torch.matmul(output_relu, self.G3(input1_4_D_reshaped).transpose(1,2))
          dot_attention_output_relu=dot_attention_output_relu.view([B,C,self.num_feat,H,W]).permute(0,2,1,3,4)
          dot_attention_output=dot_attention_output_relu
        elif act==1:
          output_sigmoid= self.sigmoid(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
          dot_attention_output_sigmoid = torch.matmul(output_sigmoid, self.G3(input1_4_D_reshaped).transpose(1,2))
          dot_attention_output_sigmoid=dot_attention_output_sigmoid.view([B,C,self.num_feat,H,W]).permute(0,2,1,3,4)
          dot_attention_output=dot_attention_output_sigmoid
        elif act==2:
          output_softmax= self.softmax(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
          dot_attention_output_softmax = torch.matmul(output_softmax, self.G3(input1_4_D_reshaped).transpose(1,2))
          dot_attention_output_softmax=dot_attention_output_softmax.view([B,C,self.num_feat,H,W]).permute(0,2,1,3,4)
          dot_attention_output=dot_attention_output_softmax
        return dot_attention_output

class SpatioTemporal_map(nn.Module):
    """ Spatiotemporal attention 1st order
    """

    def __init__(self, num_frames,num_channels_in,size,activation,act,num_feat=10,deformable_groups=8):
        super(SpatioTemporal_map, self).__init__()
        self.activation=activation
        self.num_feat=num_feat
        self.act=act
        self.conv1 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.G3 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.pool1 = torch.nn.AvgPool1d(num_feat)
        self.ffnn1 = torch.nn.Linear(num_frames*size*size, num_frames*size*size)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self,input,input1):
        B,T,C,H,W = input.shape
        act=self.act
        print(act)
        input_4_D_reshaped = input.transpose(1,2).reshape([B,C,T*H*W])
        print(act)
        if act==0:
         output_relu= self.relu(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
         map_attention_relu = torch.diag_embed(output_relu)
         map_attention_output_relu = torch.matmul(map_attention_relu, self.G3(input_4_D_reshaped).permute(0, 2, 1))
         map_attention_output_relu= map_attention_output_relu.view([B, T, H, W, self.num_feat]).permute(0, 1, 4, 2, 3)
         map_attention_output=map_attention_output_relu
        elif act==1:
         output_sigmoid= self.sigmoid(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
         map_attention_sigmoid = torch.diag_embed(output_sigmoid)
         map_attention_output_sigmoid = torch.matmul(map_attention_sigmoid, self.G3(input_4_D_reshaped).permute(0, 2, 1))
         map_attention_output_sigmoid= map_attention_output_sigmoid.view([B, T, H, W, self.num_feat]).permute(0, 1, 4, 2, 3)
         map_attention_output=map_attention_output_sigmoid
        elif act==2:
         output_softmax= self.softmax(self.ffnn1(self.pool1(self.conv1(input_4_D_reshaped).transpose(1, 2)).squeeze(-1)))
         map_attention_softmax = torch.diag_embed(output_softmax)
         map_attention_output_softmax = torch.matmul(map_attention_softmax, self.G3(input_4_D_reshaped).permute(0, 2, 1))
         map_attention_output_softmax= map_attention_output_softmax.view([B, T, H, W, self.num_feat]).permute(0, 1, 4, 2, 3)
         map_attention_output=map_attention_output_softmax
        return map_attention_output

class SpatioTemporal_dot(nn.Module):
    """ SpatioTemporal attention 2nd order
    """

    def __init__(self, num_frames,num_channels_in,size,activation,act,num_feat=10,deformable_groups=8):
        super(SpatioTemporal_dot, self).__init__()
        self.activation=activation
        self.num_feat=num_feat
        self.conv1 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.conv2 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.G3 = torch.nn.Conv1d(num_channels_in,num_feat, 1)
        self.pool1 = torch.nn.AvgPool1d(num_feat)
        self.ffnn1 = torch.nn.Linear(num_frames*size*size, num_frames*size*size)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.act=act

    def forward(self,input1,input2):
        act=self.act
        print(act)
        B,T,C,H,W = input1.shape
        dot_attention_output=0
        input1_4_D_reshaped = input1.transpose(1,2).reshape([B,C,T * H * W])
        input2_4_D_reshaped = input2.transpose(1, 2).reshape([B, C, T * H * W])
        if act==0:
         output_relu = self.relu(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
         dot_attention_output_relu= torch.matmul(output_relu, self.G3(input1_4_D_reshaped).permute(0, 2, 1))
         dot_attention_output_relu= dot_attention_output_relu.view([B, T, H, W, self.num_feat]).permute(0, 1, 4, 2, 3)
         dot_attention_output=dot_attention_output_relu
        elif act==1:
         output_sigmoid= self.sigmoid(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
         dot_attention_output_sigmoid= torch.matmul(output_sigmoid, self.G3(input1_4_D_reshaped).permute(0, 2, 1))
         dot_attention_output_sigmoid= dot_attention_output_sigmoid.view([B, T, H, W, self.num_feat]).permute(0, 1, 4, 2, 3)
         dot_attention_output=dot_attention_output_sigmoid
        elif act==2:
         output_softmax= self.softmax(torch.matmul(self.conv1(input1_4_D_reshaped).transpose(1,2),self.conv2(input2_4_D_reshaped)))
         dot_attention_output_softmax= torch.matmul(output_softmax, self.G3(input1_4_D_reshaped).permute(0, 2, 1))
         dot_attention_output_softmax= dot_attention_output_softmax.view([B, T, H, W, self.num_feat]).permute(0, 1, 4, 2, 3)
         dot_attention_output=dot_attention_output_softmax
        return dot_attention_output

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
    def __init__(self, act, nf=128, num_frames=7):
        super(EPAB_SpatioChannel, self).__init__()
        self.NL_Block_Vid_channel = SimpleNonLocal_Block_Video_NAS(num_frames, 'channel')
        #self.fusion_conv = Conv(nf * num_frames, nf, 3, 1, 1, bias=True)
        self.act=act

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

class Spatial_map(nn.Module):
    """ Spatial attention (1st order)
    """

    def __init__(self, num_frames,num_channels_in,size,activation,act,num_feat=10,deformable_groups=8):
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
        self.act=act


    def forward(self,input,input1):
        act=self.act
        B,T,C,H,W = input.shape
        input_4_D_reshaped = input.transpose(1,2).reshape([B,C,T,H*W])
        dp_all_relu=[]
        print(act)
        if act==0:
         for i in range(input_4_D_reshaped.shape[2]):
            inp=input_4_D_reshaped[:,:,i,:]
            output_relu= self.relu(self.ffnn1(self.pool1(self.conv1(inp).transpose(1, 2)).squeeze(-1)))
            map_attention_relu= torch.diag_embed(output_relu)
            map_attention_output_relu= torch.matmul(map_attention_relu, self.G3(inp).transpose(1,2))
            map_attention_output_relu=map_attention_output_relu.view([B,H,W,self.num_feat]).permute(0,3,1,2)
            dp_all_relu.append(map_attention_output_relu)
         dp_all_relu= torch.stack(dp_all_relu,dim=1)
         dp_all=dp_all_relu
        elif act==1:
         dp_all_sigmoid=[]
         for i in range(input_4_D_reshaped.shape[2]):
            inp=input_4_D_reshaped[:,:,i,:]
            output_sigmoid= self.sigmoid(self.ffnn1(self.pool1(self.conv1(inp).transpose(1, 2)).squeeze(-1)))
            map_attention_sigmoid= torch.diag_embed(output_sigmoid)
            map_attention_output_sigmoid= torch.matmul(map_attention_sigmoid, self.G3(inp).transpose(1,2))
            map_attention_output_sigmoid=map_attention_output_sigmoid.view([B,H,W,self.num_feat]).permute(0,3,1,2)
            dp_all_sigmoid.append(map_attention_output_sigmoid)
         dp_all_sigmoid= torch.stack(dp_all_sigmoid,dim=1)
         dp_all=dp_all_sigmoid
        elif act==2:
         dp_all_softmax=[]
         for i in range(input_4_D_reshaped.shape[2]):
            inp=input_4_D_reshaped[:,:,i,:]
            output_softmax= self.softmax(self.ffnn1(self.pool1(self.conv1(inp).transpose(1, 2)).squeeze(-1)))
            map_attention_softmax= torch.diag_embed(output_softmax)
            map_attention_output_softmax= torch.matmul(map_attention_softmax, self.G3(inp).transpose(1,2))
            map_attention_output_softmax=map_attention_output_softmax.view([B,H,W,self.num_feat]).permute(0,3,1,2)
            dp_all_softmax.append(map_attention_output_softmax)
         dp_all= torch.stack(dp_all_softmax,dim=1)
         #dp_all=dp_all_relu*act_weights[0]+dp_all_sigmoid*act_weights[1]+dp_all_softmax*act_weights[2]
        return dp_all

class Spatial_dot(nn.Module):
    """ Spatial attention (2nd order)
    """

    def __init__(self, num_frames,num_channels_in,size,activation,act,num_feat=10,deformable_groups=8):
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
        self.act=act

    def forward(self,input1,input2):
        B,T,C,H,W = input1.shape
        act=self.act
        print(act)
        input1_4_D_reshaped = input1.transpose(1,2).reshape([B,C,T,H*W])
        input2_4_D_reshaped = input2.transpose(1, 2).reshape([B, C,T,H*W])
        if act==0:
         dp_all_relu=[]
         for i in range(input1_4_D_reshaped.shape[2]):
            inp1=input1_4_D_reshaped[:,:,i,:]
            inp2=input2_4_D_reshaped[:,:,i,:]
            output_relu= self.relu(torch.matmul(self.conv1(inp1).transpose(1,2),self.conv2(inp2)))
            dot_attention_output_relu = torch.matmul(output_relu, self.G3(inp1).transpose(1,2))
            dot_attention_output_relu=dot_attention_output_relu.view([B,H,W,self.num_feat]).permute(0,3,1,2)
            dp_all_relu.append(dot_attention_output_relu)
         dp_all_relu=torch.stack(dp_all_relu,dim=1)
         dp_all=dp_all_relu
        elif act==1:
         dp_all_sigmoid=[]
         for i in range(input1_4_D_reshaped.shape[2]):
            inp1=input1_4_D_reshaped[:,:,i,:]
            inp2=input2_4_D_reshaped[:,:,i,:]
            output_sigmoid= self.sigmoid(torch.matmul(self.conv1(inp1).transpose(1,2),self.conv2(inp2)))
            dot_attention_output_sigmoid= torch.matmul(output_sigmoid, self.G3(inp1).transpose(1,2))
            dot_attention_output_sigmoid=dot_attention_output_sigmoid.view([B,H,W,self.num_feat]).permute(0,3,1,2)
            dp_all_sigmoid.append(dot_attention_output_sigmoid)
         dp_all_sigmoid = torch.stack(dp_all_sigmoid, dim=1)
         dp_all=dp_all_sigmoid
        elif act==2:
         dp_all_softmax=[]
         for i in range(input1_4_D_reshaped.shape[2]):
            inp1=input1_4_D_reshaped[:,:,i,:]
            inp2=input2_4_D_reshaped[:,:,i,:]
            output_softmax= self.softmax(torch.matmul(self.conv1(inp1).transpose(1,2),self.conv2(inp2)))
            dot_attention_output_softmax = torch.matmul(output_softmax, self.G3(inp1).transpose(1,2))
            dot_attention_output_softmax=dot_attention_output_softmax.view([B,H,W,self.num_feat]).permute(0,3,1,2)
            dp_all_softmax.append(dot_attention_output_softmax)
         dp_all_softmax=torch.stack(dp_all_softmax,dim=1)
         dp_all=dp_all_softmax #act_weights[0]*dp_all_relu+act_weights[1]*dp_all_sigmoid+act_weights[2]*dp_all_softmax
        return dp_all

OPS_Attention ={
    'temporal_map': lambda num_frames, num_channels_in, size,nf,act: Temporal_map(num_frames,num_channels_in,size,"relu",act,num_feat=nf),
    'temporal_dot':  lambda num_frames, num_channels_in, size,nf,act: Temporal_dot(num_frames,num_channels_in,size,"relu",act,num_feat=nf),
    'spatial_map' :  lambda num_frames, num_channels_in, size,nf,act: Spatial_map(num_frames,num_channels_in,size,"relu",act,num_feat=nf),
    'spatial_dot' :  lambda num_frames, num_channels_in, size,nf,act: Spatial_dot(num_frames,num_channels_in,size,"relu",act,num_feat=nf),
    'spatiotemporal_map': lambda num_frames, num_channels_in, size,nf,act: SpatioTemporal_map(num_frames,num_channels_in,size,"relu",act,num_feat=nf),
    'spatiotemporal_dot': lambda num_frames, num_channels_in, size,nf,act: SpatioTemporal_dot(num_frames,num_channels_in,size,"relu",act,num_feat=nf),
    'channel_map': lambda num_frames, num_channels_in, size, nf,act: Channel_map(num_frames, num_channels_in, size, "relu",act,num_feat=7),
    'channel_dot': lambda num_frames, num_channels_in, size, nf,act: Channel_dot(num_frames, num_channels_in, size, "relu",act,num_feat=7),

}