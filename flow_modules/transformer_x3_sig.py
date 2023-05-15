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
    z_m = torch.zeros(shape).type(torch.FloatTensor).cuda()
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
        self.convq1= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convq=torch.nn.Parameter(self.convq).cuda()
        nn.init.kaiming_uniform_(self.convq1, a=math.sqrt(5))
        self.convq1=torch.nn.Parameter(self.convq1,requires_grad=True).cuda()
        #z, logdet = self.attn_mask_false_elem(input=z,logdet=logdet,reverse=True,permute=False)
        self.convk1 = torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convk=torch.nn.Parameter(self.convk).cuda()
        nn.init.kaiming_uniform_(self.convk1, a=math.sqrt(5))
        self.convk1=torch.nn.Parameter(self.convk1,requires_grad=True).cuda()
        self.convq2= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convq=torch.nn.Parameter(self.convq).cuda()
        nn.init.kaiming_uniform_(self.convq2, a=math.sqrt(5))
        self.convq2=torch.nn.Parameter(self.convq2,requires_grad=True).cuda()
        #z, logdet = self.attn_mask_false_elem(input=z,logdet=logdet,reverse=True,permute=False)
        self.convk2= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convk=torch.nn.Parameter(self.convk).cuda()
        nn.init.kaiming_uniform_(self.convk2, a=math.sqrt(5))
        self.convk2=torch.nn.Parameter(self.convk2,requires_grad=True).cuda()
        self.convq3= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convq=torch.nn.Parameter(self.convq).cuda()
        nn.init.kaiming_uniform_(self.convq3, a=math.sqrt(5))
        self.convq3=torch.nn.Parameter(self.convq3,requires_grad=True).cuda()
        #z, logdet = self.attn_mask_false_elem(input=z,logdet=logdet,reverse=True,permute=False)
        self.convk3= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convk=torch.nn.Parameter(self.convk).cuda()
        nn.init.kaiming_uniform_(self.convk3, a=math.sqrt(5))
        self.convk3=torch.nn.Parameter(self.convk3,requires_grad=True).cuda()
        #self.convq4= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convq=torch.nn.Parameter(self.convq).cuda()
        #nn.init.kaiming_uniform_(self.convq4, a=math.sqrt(5))
        #self.convq4=torch.nn.Parameter(self.convq4,requires_grad=True).cuda()
        #z, logdet = self.attn_mask_false_elem(input=z,logdet=logdet,reverse=True,permute=False)
        #self.convk4= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convk=torch.nn.Parameter(self.convk).cuda()
        #nn.init.kaiming_uniform_(self.convk4, a=math.sqrt(5))
        #self.convk4=torch.nn.Parameter(self.convk4,requires_grad=True).cuda()
        #self.convq5= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convq=torch.nn.Parameter(self.convq).cuda()
        #nn.init.kaiming_uniform_(self.convq5, a=math.sqrt(5))
        #self.convq5=torch.nn.Parameter(self.convq5,requires_grad=True).cuda()
        #z, logdet = self.attn_mask_false_elem(input=z,logdet=logdet,reverse=True,permute=False)
        #self.convk5= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convk=torch.nn.Parameter(self.convk).cuda()
        #nn.init.kaiming_uniform_(self.convk5, a=math.sqrt(5))
        #self.convk5=torch.nn.Parameter(self.convk5,requires_grad=True).cuda()
        #self.convq6= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convq=torch.nn.Parameter(self.convq).cuda()
        #nn.init.kaiming_uniform_(self.convq6, a=math.sqrt(5))
        #self.convq6=torch.nn.Parameter(self.convq6,requires_grad=True).cuda()
        #z, logdet = self.attn_mask_false_elem(input=z,logdet=logdet,reverse=True,permute=False)
        #self.convk6= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convk=torch.nn.Parameter(self.convk).cuda()
        #nn.init.kaiming_uniform_(self.convk6, a=math.sqrt(5))
        #self.convk6=torch.nn.Parameter(self.convk6,requires_grad=True).cuda()
        #self.convq7= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convq=torch.nn.Parameter(self.convq).cuda()
        #nn.init.kaiming_uniform_(self.convq7, a=math.sqrt(5))
        #self.convq7=torch.nn.Parameter(self.convq7,requires_grad=True).cuda()
        #z, logdet = self.attn_mask_false_elem(input=z,logdet=logdet,reverse=True,permute=False)
        #self.convk7= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convk=torch.nn.Parameter(self.convk).cuda()
        #nn.init.kaiming_uniform_(self.convk7, a=math.sqrt(5))
        #self.convk7=torch.nn.Parameter(self.convk7,requires_grad=True).cuda()
        #self.convq8= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convq=torch.nn.Parameter(self.convq).cuda()
        #nn.init.kaiming_uniform_(self.convq8, a=math.sqrt(5))
        #self.convq8=torch.nn.Parameter(self.convq8,requires_grad=True).cuda()
        #z, logdet = self.attn_mask_false_elem(input=z,logdet=logdet,reverse=True,permute=False)
        #self.convk8= torch.empty([num_channels,num_channels,1,1]).type(torch.FloatTensor).cuda()
        #self.convk=torch.nn.Parameter(self.convk).cuda()
        #nn.init.kaiming_uniform_(self.convk8, a=math.sqrt(5))
        #self.convk8=torch.nn.Parameter(self.convk8,requires_grad=True).cuda()
        #self.convk7=torch.nn.Parameter(self.convk7,requires_grad=True).cuda()
        #self.off= torch.nn.Parameter(torch.ones([1],dtype=torch.double)*(0.99),requires_grad=True).cuda()#(torch.nn.Parameter(torch.ones([1])*(0.99)).type(torch.DoubleTensor)).cuda()
        self.register_parameter("offset",nn.Parameter(torch.ones([1,1,1])*0.99))
        #self.off2= torch.nn.Parameter(torch.ones([1],dtype=torch.double)*(12),requires_grad=True).cuda()
        #self.register_parameter("offset2", self.off2)
        self.register_parameter("offset2",nn.Parameter(torch.ones([1,1,1])*0.65))
        #self.off3= torch.nn.Parameter(torch.ones([1],dtype=torch.double)*(6),requires_grad=True).cuda()
        self.register_parameter("offset3",nn.Parameter(torch.ones([1,1,1])*-0.6))
        self.register_parameter("scale",nn.Parameter(torch.ones([1,1,1])*100))
        #self.register_parameter("offset3", self.off3)
    def masked_softmax(self, vec, mask, dim=-1, epsilon=1e-5):
        exps = torch.exp(vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps/masked_sums)
    def forward(self, input: torch.Tensor, logdet=0, reverse=False, permute=False):
     if not reverse:
        z=input
        p =z.shape[-1]//2  #path size
        #scale=np.sqrt(self.c*p*p)
        #print(self.scale)
        #print(self.offset)
        full_inp = rearrange(z, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p) #get patches 
        mask = torch.Tensor(checkerboard(full_inp[0, :, :].shape)).type(torch.FloatTensor).cuda()
        if permute:
           mask=1-mask
        f_m = full_inp * mask
        num_patches_sqrt = z.shape[-1] // p # Get sqrt of number of patches 
        z_m = reverse_rearrange(f_m, p, num_patches_sqrt ** 2, z.shape)#masked input
        q1= torch.nn.functional.conv2d(z_m,self.convq1)#Convq
        q2= torch.nn.functional.conv2d(z_m,self.convq2)#Convq
        q3= torch.nn.functional.conv2d(z_m,self.convq3)#Convq
        #q4= torch.nn.functional.conv2d(z_m,self.convq4)#Convq
        #q5= torch.nn.functional.conv2d(z_m,self.convq5)#Convq
        #q6= torch.nn.functional.conv2d(z_m,self.convq6)#Convq
        #q7= torch.nn.functional.conv2d(z_m,self.convq7)#Convq
        #q8= torch.nn.functional.conv2d(z_m,self.convq8)#Convq
        k1= torch.nn.functional.conv2d(z_m,self.convk1)#Convk
        k2= torch.nn.functional.conv2d(z_m,self.convk2)#Convk
        k3= torch.nn.functional.conv2d(z_m,self.convk3)#Convk
        #k4= torch.nn.functional.conv2d(z_m,self.convk4)#Convk
        #k5= torch.nn.functional.conv2d(z_m,self.convk5)#Convk
        #k6= torch.nn.functional.conv2d(z_m,self.convk6)#Convk
        #k7= torch.nn.functional.conv2d(z_m,self.convk7)#Convk
        #k8= torch.nn.functional.conv2d(z_m,self.convk8)#Convk
        s = nn.Sigmoid()
        full_inp_q1 = (rearrange(q1, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))#get patches from q
        full_inp_q2 = (rearrange(q2, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))#get patches from q
        full_inp_q3= (rearrange(q3, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))#get patches from q
        #full_inp_q4= (rearrange(q4, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))#get patches from q
        #full_inp_q5= (rearrange(q5, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))#get patches from q
        #full_inp_q6= (rearrange(q6, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))#get patches from q
        #full_inp_q7= (rearrange(q7, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))#get patches from q
        #full_inp_q8= (rearrange(q8, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))#get patches from q
        full_inp_k1 = (rearrange(k1, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))##get patches from k
        full_inp_k2= (rearrange(k2, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))##get patches from k
        full_inp_k3= (rearrange(k3, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))##get patches from k
        #full_inp_k4= (rearrange(k4, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))##get patches from k
        #full_inp_k5= (rearrange(k5, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))##get patches from k
        #full_inp_k6= (rearrange(k6, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))##get patches from k
        #full_inp_k7= (rearrange(k7, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))##get patches from k
        #full_inp_k8= (rearrange(k8, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))##get patches from k
        s = nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        attn_mask = torch.Tensor(checkerboard([full_inp_q1.shape[1],full_inp_q1.shape[1]])).type(torch.FloatTensor).cuda()
        #if permute:
        #  attn_mask=1-attn_mask
        #attn=self.masked_softmax((torch.matmul(full_inp_q, full_inp_k.permute(0, 2,1))/self.scale),attn_mask)
        attn = (s((torch.matmul(full_inp_q1, full_inp_k1.permute(0, 2,
            1))/self.scale)+(torch.matmul(full_inp_q2, full_inp_k2.permute(0, 2,
            1))/self.scale)+(torch.matmul(full_inp_q3, full_inp_k3.permute(0, 2,
            1))/self.scale)+self.offset2)+self.offset3)*attn_mask #Compute attention (masked) attn_mask =
        #torch.Tensor(checkerboard(attn[0, :,
        #    :].shape)).type(torch.FloatTensor).cuda()
        #if permute:
        #   attn_mask=1-attn_mask
        #print(attn)
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
        #scale=np.sqrt(self.c*p*p)
        out_final = rearrange(z, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        num_patches_sqrt = z.shape[-1] // p
        s = nn.Sigmoid()
        #Recompute attention 
        mask = torch.Tensor(checkerboard(out_final[0, :, :].shape)).type(torch.FloatTensor).cuda()
        if permute:
           mask=1-mask
        rev = out_final * (mask)
        rev_rearrange = reverse_rearrange(rev, p, num_patches_sqrt ** 2, z.shape)
        q1= torch.nn.functional.conv2d(rev_rearrange,self.convq1)
        q2= torch.nn.functional.conv2d(rev_rearrange,self.convq2)
        q3= torch.nn.functional.conv2d(rev_rearrange,self.convq3)
        #q4= torch.nn.functional.conv2d(rev_rearrange,self.convq4)
        #q5= torch.nn.functional.conv2d(rev_rearrange,self.convq5)
        #q6= torch.nn.functional.conv2d(rev_rearrange,self.convq6)
        #q7= torch.nn.functional.conv2d(rev_rearrange,self.convq7)
        #q8= torch.nn.functional.conv2d(rev_rearrange,self.convq8)
        k1= torch.nn.functional.conv2d(rev_rearrange,self.convk1)
        k2= torch.nn.functional.conv2d(rev_rearrange,self.convk2)
        k3= torch.nn.functional.conv2d(rev_rearrange,self.convk3)
        #k4= torch.nn.functional.conv2d(rev_rearrange,self.convk4)
        #k5= torch.nn.functional.conv2d(rev_rearrange,self.convk5)
        #k6= torch.nn.functional.conv2d(rev_rearrange,self.convk6)
        #k7= torch.nn.functional.conv2d(rev_rearrange,self.convk7)
        #k8= torch.nn.functional.conv2d(rev_rearrange,self.convk8)
        full_inp_q1_rev = (rearrange(q1, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        full_inp_q2_rev = (rearrange(q2, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        full_inp_q3_rev = (rearrange(q3, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        #full_inp_q4_rev = (rearrange(q4, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        #full_inp_q5_rev = (rearrange(q5, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        #full_inp_q6_rev = (rearrange(q6, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        #full_inp_q7_rev = (rearrange(q7, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        #full_inp_q8_rev = (rearrange(q8, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        full_inp_k1_rev = (rearrange(k1, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        full_inp_k2_rev = (rearrange(k2, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        full_inp_k3_rev = (rearrange(k3, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        #full_inp_k4_rev = (rearrange(k4, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        #full_inp_k5_rev = (rearrange(k5, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        #full_inp_k6_rev = (rearrange(k6, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        #full_inp_k7_rev = (rearrange(k7, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        #full_inp_k8_rev = (rearrange(k8, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        attn_mask = torch.Tensor(checkerboard([full_inp_q1_rev.shape[1], full_inp_q1_rev.shape[1]])).type(torch.FloatTensor).cuda()
        #if permute:
        #   attn_mask=1-attn_mask
        #attn=self.masked_softmax((torch.matmul(full_inp_q_rev, full_inp_k_rev.permute(0, 2,1))/self.scale),attn_mask)
        attn = (s((torch.matmul(full_inp_q1_rev, full_inp_k1_rev.permute(0, 2, 1))/self.scale)+(torch.matmul(full_inp_q2_rev, full_inp_k2_rev.permute(0, 2, 1))/self.scale)+(torch.matmul(full_inp_q3_rev, full_inp_k3_rev.permute(0, 2, 1))/self.scale)+self.offset2)+self.offset3)*attn_mask
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
