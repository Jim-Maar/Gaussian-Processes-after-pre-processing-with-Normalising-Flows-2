import torch
from torch import nn as nn
import math
import torch.nn.functional as F
import torch
from einops import rearrange
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
def checkerboard(shape):
    return 1 - np.indices(shape).sum(axis=0) % 2


def reverse_rearrange(x, patch_size, num_patch, shape):
    # print(shape)
    z_m = torch.zeros(shape).type(torch.DoubleTensor).cuda()
    # print(z_m.shape)
    # print(x.shape)
    start_h = 0
    start_w = 0
    for i in range(0, num_patch):
        z_m[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size] = x[:, i, :].view(
            [shape[0], shape[1], patch_size, patch_size])
        start_w = start_w + patch_size
        if start_w == shape[-1]:
            start_h = start_h + patch_size
            start_w = 0

    return z_m

class Transformer_attn(nn.Module):
    def __init__(self, num_channels):
        super(Transformer_attn,self).__init__()
        self.c=num_channels
        w_shape = [num_channels, num_channels]
        self.convq = torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convq=torch.nn.Parameter(self.convq).cuda()
        nn.init.kaiming_uniform_(self.convq, a=math.sqrt(5))
        self.convq=torch.nn.Parameter(self.convq,requires_grad=True).cuda()
        #z, logdet = self.attn_mask_false_elem(input=z,logdet=logdet,reverse=True,permute=False)
        self.convk = torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convk=torch.nn.Parameter(self.convk).cuda()
        nn.init.kaiming_uniform_(self.convk, a=math.sqrt(5))
        self.convk=torch.nn.Parameter(self.convk,requires_grad=True).cuda()
        #self.off= torch.nn.Parameter(torch.ones([1],dtype=torch.double)*(0.99),requires_grad=True).cuda()#(torch.nn.Parameter(torch.ones([1])*(0.99)).type(torch.DoubleTensor)).cuda()
        self.register_parameter("offset",nn.Parameter(torch.ones([1,1,1])*1.5))
        #self.off2= torch.nn.Parameter(torch.ones([1],dtype=torch.double)*(12),requires_grad=True).cuda()
        #self.register_parameter("offset2", self.off2)
        self.register_parameter("offset2",nn.Parameter(torch.ones([1,1,1])*0.65))
        #self.off3= torch.nn.Parameter(torch.ones([1],dtype=torch.double)*(6),requires_grad=True).cuda()
        self.register_parameter("offset3",nn.Parameter(torch.ones([1,1,1])*-0.6))
        #self.register_parameter("offset3", self.off3)
    def forward(self, input: torch.Tensor, logdet=0, reverse=False, permute=False):
     if not reverse:
        z=input
        p =z.shape[-1]//2  #path size
        scale=np.sqrt(self.c*p*p)
        full_inp = rearrange(z, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p) #get patches 
        mask = torch.Tensor(checkerboard(full_inp[0, :, :].shape)).type(torch.FloatTensor).cuda()
        if permute:
           mask=1-mask
        f_m = full_inp * mask
        num_patches_sqrt = z.shape[-1] // p # Get sqrt of number of patches 
        z_m = reverse_rearrange(f_m, p, num_patches_sqrt ** 2, z.shape)#masked input
        q= torch.nn.functional.conv2d(z_m,self.convq)#Convq
        k= torch.nn.functional.conv2d(z_m,self.convk)#Convk
        s = nn.Sigmoid()
        full_inp_q = (rearrange(q, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))#get patches from q
        full_inp_k = (rearrange(k, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))##get patches from k
        s = nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        attn_mask = torch.Tensor(checkerboard([full_inp_q.shape[1],full_inp_q.shape[1]])).type(torch.FloatTensor).cuda()
        #if permute:
        #  attn_mask=1-attn_mask
        attn = (s((torch.matmul(full_inp_q, full_inp_k.permute(0, 2, 1))/scale)+self.offset2)+self.offset3)*attn_mask #Compute attention (masked)
        attn_mask = torch.Tensor(checkerboard(attn[0, :, :].shape)).type(torch.FloatTensor).cuda()
        #if permute:
        #   attn_mask=1-attn_mask
        half = attn.shape[-1] // 2
        det_attn = attn[:, attn_mask > 0].view(attn.shape[0], attn.shape[-1], half)
        mask2 = torch.Tensor(checkerboard([attn.shape[1]])).type(torch.FloatTensor).cuda()
        #Compute log abs determinant
        #if permute:
        #    mask2=1-mask2
        m1 = det_attn[:, mask2 > 0, :]#+f
        m2 = det_attn[:, (1 - mask2) > 0, :]#+f
        f=torch.eye(m1.shape[-1]).cuda()*(self.offset)
        m1=m1+f
        m2=m2+f
        det_attn[:, mask2 > 0, :] = m1
        det_attn[:, (1 - mask2) > 0, :] =m2
        attn[:, attn_mask > 0] = det_attn.view(attn.shape[0],-1) # Change attention (after adding f to m1 and m2)
        a = torch.slogdet(m1)[1]*p*(p//2)*self.c
        b = torch.slogdet(m2)[1]*p*(p//2)*self.c
        logdet = logdet +(a+b)#+10
        #Compute attended input(masked)
        out = torch.matmul(attn, full_inp * (1 - mask))
        out_final = out * (1 - mask) + full_inp * mask
        z_out = reverse_rearrange(out_final, p, num_patches_sqrt ** 2, z.shape)
        output=z_out


     else:
        z=input
        #print("Offset")
        #print(self.offset)
        #print(self.offset2)
        #print(self.offset3)
        p = z.shape[-1]//2
        scale=np.sqrt(self.c*p*p)
        out_final = rearrange(z, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        num_patches_sqrt = z.shape[-1] // p
        s = nn.Sigmoid()
        #Recompute attention 
        mask = torch.Tensor(checkerboard(out_final[0, :, :].shape)).type(torch.FloatTensor).cuda()
        if permute:
           mask=1-mask
        rev = out_final * (mask)
        rev_rearrange = reverse_rearrange(rev, p, num_patches_sqrt ** 2, z.shape)
        q= torch.nn.functional.conv2d(rev_rearrange,self.convq)
        k= torch.nn.functional.conv2d(rev_rearrange,self.convk)
        full_inp_q_rev = (rearrange(q, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        full_inp_k_rev = (rearrange(k, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        attn_mask = torch.Tensor(checkerboard([full_inp_q_rev.shape[1], full_inp_q_rev.shape[1]])).type(torch.FloatTensor).cuda()
        #if permute:
        #   attn_mask=1-attn_mask
        attn = (s((torch.matmul(full_inp_q_rev, full_inp_k_rev.permute(0, 2, 1))/scale)+self.offset2)+self.offset3)*attn_mask
        attn_mask = torch.Tensor(checkerboard(attn[0, :, :].shape)).cuda()
        #if permute:
        #   attn_mask=1-attn_mask
        half = attn.shape[-1] // 2
        det_attn = attn[:, attn_mask > 0].view(attn.shape[0], attn.shape[-1], half)
        mask2 = torch.Tensor(checkerboard([attn.shape[1]])).type(torch.FloatTensor).cuda()
        #if permute:
        #    mask2=1-mask2
        m1 = det_attn[:, mask2 > 0, :]#+f
        m2 = det_attn[:, (1 - mask2) > 0, :]#+f
        f=torch.eye(m1.shape[-1]).cuda()*self.offset
        m1=m1+f
        m2=m2+f
        det_attn[:, mask2 > 0, :] = m1
        det_attn[:, (1 - mask2) > 0, :] =m2

        attn[:, attn_mask > 0] = det_attn.view(attn.shape[0],-1)# Change attention (after adding f to m1 and m2)
        #Compute reverse pass 
        mask2 = torch.Tensor(checkerboard([attn.shape[1]])).type(torch.FloatTensor).cuda()
        #if permute:
        #    mask2=1-mask2
        m1_inv= torch.inverse(m1)
        m2_inv= torch.inverse(m2)
        rev_unmask = out_final * (1 - mask)
        rev_mask = torch.Tensor(checkerboard(rev_unmask[0, :, :].shape)).type(torch.FloatTensor).cuda()
        if permute:
           rev_mask=1-rev_mask
        full_rev = rev_unmask[:, (1 - rev_mask) > 0].view(rev_unmask.shape[0], rev_unmask.shape[1],
                                                      rev_unmask.shape[-1] // 2)
        mask2 = torch.Tensor(checkerboard([rev_unmask.shape[1]]))#.cuda()
        #if permute:
        #    mask2=1-mask2
        #if permute:
        #    mask2=1-mask2
        m1_2 = full_rev[:, mask2 > 0, :]
        m2_2 = full_rev[:, (1 - mask2) > 0, :]
        mask = torch.Tensor(checkerboard(out_final[0, :, :].shape)).type(torch.FloatTensor).cuda()
        if permute:
           mask=1-mask
        mask2 = torch.Tensor(checkerboard([rev.shape[1], ])).type(torch.FloatTensor).cuda()
        #if permute:
        #    mask2=1-mask2
        mask = 1 - mask
        mask[(1 - mask2) > 0, :] = 0
        rev[:, mask > 0] = torch.matmul(m1_inv, m1_2).view(rev.shape[0], -1)
        mask = torch.Tensor(checkerboard(out_final[0, :, :].shape)).type(torch.FloatTensor).cuda()
        mask = 1 - mask
        if permute:
           mask=1-mask
        mask[mask2 > 0, :] = 0
        rev[:, mask > 0] = torch.matmul(m2_inv, m2_2).view(rev.shape[0], -1) #Find solutions of simultaneous equations Ax=b
        output=reverse_rearrange(rev, p, num_patches_sqrt ** 2, z.shape)
        half = attn.shape[-1] // 2
        det_attn = attn[:, attn_mask > 0].view(attn.shape[0], attn.shape[-1], half)
        mask2 = torch.Tensor(checkerboard([attn.shape[1]])).type(torch.FloatTensor).cuda()
        #Compute reverse logdet
        #if permute:
        #    mask2=1-mask2
        m1 = det_attn[:, mask2 > 0, :]
        m2 = det_attn[:, (1 - mask2) > 0, :]
        a = torch.slogdet(m1)[1] *p*(p//2)*self.c
        b = torch.slogdet(m2)[1] *p*(p//2)*self.c
        logdet = logdet - (a+b)#-10
     return output,logdet
