import torch
#x=torch.randn([1,3,8,8],requires_grad=True)
import torch.nn as nn
import numpy as np
import torch
import math

class Elementwise_channel_exp(nn.Module):
  def __init__(self, num_channels):
    super().__init__()
    #w_shape = [num_channels, num_channels]
    #w_init = np.random.randn(num_channels,num_channels,1,1).astype(np.float32)
    #self.weight=torch.nn.Parameter(torch.Tensor(w_init)).double().type(torch.DoubleTensor).cuda()
    w_shape = [num_channels, num_channels]
    self.weight = torch.empty([num_channels,num_channels,1,1]).type(torch.DoubleTensor).cuda()
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #self.convq = torch.empty([num_channels,num_channels,1,1]).type(torch.DoubleTensor).cuda()
    #self.convq=torch.nn.Parameter(self.convq).cuda()
    #nn.init.kaiming_uniform_(self.convq, a=math.sqrt(5))
    #self.convq=torch.nn.Parameter(self.convq,requires_grad=True).cuda()
    self.weight=torch.nn.Parameter(self.weight,requires_grad=True).cuda()
    self.channel_offset = torch.ones([1,num_channels,1,1]).type(torch.DoubleTensor).cuda()*8
    #nn.init.kaiming_uniform_(self.channel_offset, a=math.sqrt(5))
    self.channel_offset =torch.nn.Parameter(self.channel_offset,requires_grad=True).cuda()
  def init_mask(self,shape):
    #print(shape)
    self.shape=shape
    self.mask = torch.DoubleTensor(shape[0],shape[1],shape[2]).uniform_() > 0.5
    self.mask= self.mask.int().cuda()
  def forward(self,input: torch.Tensor, logdet=0, reverse=False,permute=False):
   if not reverse:
    print(self.shape)
    z=input
    #torch.Tensor(checkerboard(z[0].shape)).double().cuda()
    mask=self.mask
    #print(mask.shape)
    #print(z.shape)
    input_mask=z*mask
    s=torch.nn.Sigmoid()
    out=s(torch.nn.functional.conv2d(input_mask,self.weight)+self.channel_offset)+0.001
    out_final=z*(1-mask)*out+z*mask
    #out_m=torch.nn.functional.conv2d(input_mask,self.weight)
    attn_used=out[:,(1-mask)!=0]
    #print(attn_used.shape)
    dlogdet=torch.sum(torch.log(attn_used),dim=-1)
    #print("Dlogdet attn",dlogdet)
    logdet=logdet+dlogdet
    return out_final,logdet
   else:
    z=input.double()
    mask=self.mask
    #mask=torch.Tensor(checkerboard(z[0].shape)).double()#.cuda()
    #if permute:
    #    mask=1-mask
    s=torch.nn.Sigmoid()
    attn_rev=s(torch.nn.functional.conv2d(z*mask,self.weight)+self.channel_offset)+0.001
    input_rev=torch.div(z*(1-mask),attn_rev.double())*(1-mask)+z*mask
    input_mask=input_rev*mask
    s=torch.nn.Sigmoid()
    out=s(torch.nn.functional.conv2d(input_mask,self.weight)+self.channel_offset)+0.001
    out_m=torch.nn.functional.conv2d(input_mask,self.weight)
    attn_used=out[:,(1-mask)!=0]
    dlogdet=torch.sum(torch.log(attn_used),dim=-1)
    logdet=logdet-dlogdet
    return input_rev,logdet
