import torch
from torch import nn as nn
import math
import torch.nn.functional as F
import torch
from einops import rearrange
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(threshold=10_000)
inp = torch.randn([4, 3, 16, 16], requires_grad=True)

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = 3 * 80 * 80  # thops.pixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float() \
                    .view(w_shape[0], w_shape[1], 1, 1)
            # print(dlogdet)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = thops.sum(self.log_s) * thops.pixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        # print(weight)
        # weight=self.weight
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            # print(dlogdet)
            return z#, dlogdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z#, logdet


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
        super().__init__()
        self.c=num_channels
        w_shape = [num_channels, num_channels]
        #w_init1 = np.random.randn(num_channels,num_channels,1,1).astype(np.float32)
        #w_init2 = np.random.randn(num_channels,num_channels,1,1).astype(np.float32)
        #self.convq = torch.nn.Parameter(torch.Tensor(w_init1)).type(torch.DoubleTensor).cuda() #InvertibleConv1x1(self.c).double()
        #self.convk =  torch.nn.Parameter(torch.Tensor(w_init2)).type(torch.DoubleTensor).cuda()#InvertibleConv1x1(self.c).double()
        w_shape = [num_channels, num_channels]
        self.convq = torch.empty([num_channels,num_channels,1,1]).type(torch.DoubleTensor).cuda()
        nn.init.kaiming_uniform_(self.convq, a=math.sqrt(5))
        self.convk = torch.empty([num_channels,num_channels,1,1]).type(torch.DoubleTensor).cuda()
        nn.init.kaiming_uniform_(self.convk, a=math.sqrt(5))
        self.offset= (torch.nn.Parameter(torch.ones([1])*(1.001)).type(torch.DoubleTensor)).cuda()
        self.offset2= (torch.nn.Parameter(torch.ones([1])*(10)).type(torch.DoubleTensor)).cuda()
    def forward(self, input: torch.Tensor, logdet=0, reverse=False, ft=None):
     if not reverse:
        #logdet=0
        #input=z
        z=input
        p =   2 #z.shape[-1]//2
        #print("Patch size",p)
        full_inp = rearrange(z, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        mask = torch.Tensor(checkerboard(full_inp[0, :, :].shape)).type(torch.DoubleTensor).cuda()
        f_m = full_inp * mask
        num_patches_sqrt = z.shape[-1] // p
        z_m = reverse_rearrange(f_m, p, num_patches_sqrt ** 2, z.shape)
        q= torch.nn.functional.conv2d(z_m,self.convq)
        k= torch.nn.functional.conv2d(z_m,self.convk)
        full_inp_q = rearrange(q, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        full_inp_k = rearrange(k, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        s = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        attn_mask = torch.Tensor(checkerboard([full_inp_q.shape[1],full_inp_q.shape[1]])).type(torch.DoubleTensor).cuda()
        attn = (s(torch.matmul(full_inp_q, full_inp_k.permute(0, 2, 1))+self.offset2)+0.0001)*attn_mask
        #attn = s(torch.matmul(full_inp_q, full_inp_k.permute(0, 2, 1)))
        attn_mask = torch.Tensor(checkerboard(attn[0, :, :].shape)).type(torch.DoubleTensor).cuda()
        half = attn.shape[-1] // 2
        #print(attn)
        det_attn = attn[:, attn_mask > 0].view(attn.shape[0], attn.shape[-1], half)
        mask2 = torch.Tensor(checkerboard([attn.shape[1]])).type(torch.DoubleTensor).cuda()
        #f=torch.eye(m1.shape[-1]).cuda()*(1.001)
        m1 = det_attn[:, mask2 > 0, :]#+f
        m2 = det_attn[:, (1 - mask2) > 0, :]#+f
        f=torch.eye(m1.shape[-1]).cuda()*(self.offset)
        m1=m1+f
        m2=m2+f
        det_attn[:, mask2 > 0, :] = m1
        det_attn[:, (1 - mask2) > 0, :] =m2
        #print(attn[:, attn_mask > 0].shape)
        #print(det_attn.shape)
        attn[:, attn_mask > 0] = det_attn.view(attn.shape[0],-1)
        a = torch.slogdet(m1)[1]*p*(p//2)*self.c
        b = torch.slogdet(m2)[1]*p*(p//2)*self.c
        logdet = logdet +(a+b)#*p*p//2*#*0.000001

        #print(logdet)
        out = torch.matmul(attn, full_inp * (1 - mask))
        out_final = out * (1 - mask) + full_inp * mask
        z_out = reverse_rearrange(out_final, p, num_patches_sqrt ** 2, z.shape)
        output=z_out

     else:
        z=input
        p = 2 #z.shape[-1]//2
        out_final = rearrange(z, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        num_patches_sqrt = z.shape[-1] // p
        mask = torch.Tensor(checkerboard(out_final[0, :, :].shape)).type(torch.DoubleTensor).cuda()
        rev = out_final * (mask)
        rev_rearrange = reverse_rearrange(rev, p, num_patches_sqrt ** 2, z.shape)
        q= torch.nn.functional.conv2d(rev_rearrange,self.convq)
        k= torch.nn.functional.conv2d(rev_rearrange,self.convk)
        full_inp_q_rev = rearrange(q, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        full_inp_k_rev = rearrange(k, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        s = torch.nn.Sigmoid()
        #f=torch.eye(m1.shape[-1])*(1.001)
        attn_mask = torch.Tensor(checkerboard([full_inp_q_rev.shape[1], full_inp_q_rev.shape[1]])).type(torch.DoubleTensor).cuda()
        attn = (s(torch.matmul(full_inp_q_rev, full_inp_k_rev.permute(0, 2, 1))+self.offset2)+0.0001)*attn_mask
        #print(attn)
        attn_mask = torch.Tensor(checkerboard(attn[0, :, :].shape)).cuda()
        half = attn.shape[-1] // 2
        det_attn = attn[:, attn_mask > 0].view(attn.shape[0], attn.shape[-1], half)
        mask2 = torch.Tensor(checkerboard([attn.shape[1]])).type(torch.DoubleTensor).cuda()
        m1 = det_attn[:, mask2 > 0, :]#+f
        m2 = det_attn[:, (1 - mask2) > 0, :]#+f
        f=torch.eye(m1.shape[-1]).cuda()*(self.offset)
        m1=m1+f
        m2=m2+f
        det_attn[:, mask2 > 0, :] = m1
        det_attn[:, (1 - mask2) > 0, :] =m2

        attn[:, attn_mask > 0] = det_attn.view(attn.shape[0],-1)
        #print(attn)
        mask2 = torch.Tensor(checkerboard([attn.shape[1]])).type(torch.DoubleTensor).cuda()
        m1_inv= torch.inverse(m1.double())
        m2_inv= torch.inverse(m2.double())
        rev_unmask = out_final * (1 - mask)
        rev_mask = torch.Tensor(checkerboard(rev_unmask[0, :, :].shape)).type(torch.DoubleTensor).cuda()
        full_rev = rev_unmask[:, (1 - rev_mask) > 0].view(rev_unmask.shape[0], rev_unmask.shape[1],
                                                      rev_unmask.shape[-1] // 2)
        mask2 = torch.Tensor(checkerboard([rev_unmask.shape[1]]))#.cuda()
        m1_2 = full_rev[:, mask2 > 0, :]
        m2_2 = full_rev[:, (1 - mask2) > 0, :]
        mask = torch.Tensor(checkerboard(out_final[0, :, :].shape)).type(torch.DoubleTensor).cuda()
        mask2 = torch.Tensor(checkerboard([rev.shape[1], ])).type(torch.DoubleTensor).cuda()
        mask = 1 - mask
        mask[(1 - mask2) > 0, :] = 0
        rev[:, mask > 0] = torch.matmul(m1_inv, m1_2).view(rev.shape[0], -1)
        mask = torch.Tensor(checkerboard(out_final[0, :, :].shape)).type(torch.DoubleTensor).cuda()
        mask = 1 - mask
        mask[mask2 > 0, :] = 0
        rev[:, mask > 0] = torch.matmul(m2_inv, m2_2).view(rev.shape[0], -1)
        output=reverse_rearrange(rev, p, num_patches_sqrt ** 2, z.shape)
        half = attn.shape[-1] // 2
        det_attn = attn[:, attn_mask > 0].view(attn.shape[0], attn.shape[-1], half)
        mask2 = torch.Tensor(checkerboard([attn.shape[1]])).type(torch.DoubleTensor).cuda()
        m1 = det_attn[:, mask2 > 0, :]
        m2 = det_attn[:, (1 - mask2) > 0, :]
        a = torch.slogdet(m1)[1] *p*(p//2)*self.c
        b = torch.slogdet(m2)[1] *p*(p//2)*self.c
        logdet = logdet - (a+b)
     return output,logdet
import torch
#x=torch.randn([1,3,8,8],requires_grad=True)
import torch.nn as nn
import numpy as np
import torch

def checkerboard(shape):
    return 1-np.indices(shape).sum(axis=0) % 2
mask=torch.Tensor(checkerboard([1,3,8,8]))

class Elementwise_channel(nn.Module):
  def __init__(self, num_channels):
    super().__init__()
    w_shape = [num_channels, num_channels]
    w_init = np.random.randn(num_channels,num_channels,1,1).astype(np.float32)
    self.weight=torch.nn.Parameter(torch.Tensor(w_init)).cuda()
  def forward(self, z: torch.Tensor, logdet=0, reverse=False, permute=False):
   if not reverse:
    mask=torch.Tensor(checkerboard(z[0].shape)).cuda()
    if permute:
       mask=1-mask
    input_mask=z*mask
    s=torch.nn.Sigmoid()
    out=s(torch.nn.functional.conv2d(input_mask,self.weight))
    out_final=z*(1-mask)*out+z*mask
    attn_used=out[:,(1-mask)!=0]
    dlogdet=torch.sum(torch.log(torch.abs(attn_used)),dim=-1)
    logdet=logdet+dlogdet
    return out_final,logdet
   else:
    mask=torch.Tensor(checkerboard(z[0].shape)).cuda()
    if permute:
       mask=1-mask
    s=torch.nn.Sigmoid()
    attn_rev=s(torch.nn.functional.conv2d(z*mask,self.weight))
    input_rev=((z*(1-mask))/attn_rev)*(1-mask)+z*mask 
    input_mask=input_rev*mask
    s=torch.nn.Sigmoid()
    out=s(torch.nn.functional.conv2d(input_mask,self.weight))
    attn_used=out[:,(1-mask)!=0]
    dlogdet=torch.sum(torch.log(torch.abs(attn_used)),dim=-1)
    logdet=logdet-dlogdet
    return input_rev,logdet

class Elementwise_channel_exp(nn.Module):
  def __init__(self, num_channels):
    super().__init__()
    #w_shape = [num_channels, num_channels]
    #w_init = np.random.randn(num_channels,num_channels,1,1).astype(np.float32)
    #self.weight=torch.nn.Parameter(torch.Tensor(w_init)).double().type(torch.DoubleTensor).cuda()
    w_shape = [num_channels, num_channels]
    self.weight = torch.empty([num_channels,num_channels,1,1]).cuda()
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    self.weight=torch.nn.Parameter(self.weight.type(torch.DoubleTensor)).cuda()
    self.channel_offset = torch.ones([1,num_channels,1,1]).cuda()
    #nn.init.kaiming_uniform_(self.channel_offset, a=math.sqrt(5))
    self.channel_offset =torch.nn.Parameter(self.channel_offset.type(torch.DoubleTensor)).cuda()
  def forward(self,input: torch.Tensor, logdet=0, reverse=False,permute=False):
   if not reverse:
    z=input
    mask=torch.Tensor(checkerboard(z[0].shape)).double().cuda()
    if permute:
        mask=1-mask
    input_mask=z*mask
    s=torch.nn.Sigmoid()
    out=s(torch.nn.functional.conv2d(input_mask,self.weight)+self.channel_offset)+0.001
    out_final=z*(1-mask)*out+z*mask
    #out_m=torch.nn.functional.conv2d(input_mask,self.weight)
    attn_used=out[:,(1-mask)!=0]
    #print(attn_used.shape)
    dlogdet=torch.sum(torch.log(attn_used),dim=-1)
    print("Dlogdet attn",dlogdet)
    logdet=logdet+dlogdet
    return out_final.float(),logdet
   else:
    z=input
    mask=torch.Tensor(checkerboard(z[0].shape)).double().cuda()
    if permute:
        mask=1-mask
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
    return input_rev.float(),logdet

import math
import torch.nn.functional as F
'''class _Spatial_first_order_attn(nn.Module):
    def __init__(self, input_channels):
        super(_Spatial_first_order_attn, self).__init__()
        #self.num_channels=num_channels
        self.input_channels=input_channels
        self.weight = torch.empty([self.input_channels,self.input_channels,1]).cuda()
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight=torch.nn.Parameter(self.weight.type(torch.DoubleTensor)).cuda()
        self.bias = torch.empty([self.input_channels]).cuda()
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.bias =torch.nn.Parameter(self.bias.type(torch.DoubleTensor)).cuda()
        self.s = torch.nn.Parameter(torch.randn([1,self.input_channels,1]).type(torch.DoubleTensor)).cuda()
        self.pool1 = torch.nn.AvgPool1d(self.input_channels)

    def init_mask(self,permute=False):
        mask = torch.zeros((1, self.num_channels,self.input_channels)).double().cuda()
        mask[:, 1::2, ::2] = 1
        mask[:, ::2, 1::2] = 1
        mask = mask.permute(0, 2, 1)
        if permute:
            mask=1-mask
        self.mask=mask

    def forward(self, input: torch.Tensor, logdet=0, reverse=False, permute=False):
        # print(input)
        # print(weight)
      if not reverse:
        self.num_channels=input.shape[-1]**2
        self.init_mask(permute)
        input = input
        B, C, H, W = input.shape

        sig = torch.nn.Sigmoid()
        input_masked = input.view(B, C, H * W) * self.mask
        z = F.conv1d(input_masked, self.weight, bias=self.bias)
        z_new = z.transpose(1, 2)
        pool_out = self.pool1(z_new)
        attn_out = (sig(pool_out.squeeze(2)+2)+0.0001).unsqueeze(1)
        attn_mask = (1 - self.mask) * attn_out + self.mask * (sig(self.s+2)+0.0001)
        out_new = input * attn_mask.view(B, C,H * W).view(B, C, H, W)
        log_det_final = 0
        #print(pool_out.shape[0])
        dets=[]
        for i in range(pool_out.shape[0]):
            log_det = 0
            for j in range(pool_out.shape[-2]):
                scale = pool_out[i, j, 0]
                log_det = log_det + (self.input_channels// 2) * torch.log(sig(scale+2)+0.0001)
            #print(log_det.shape)
            log_det = log_det + torch.sum(torch.log(sig(self.s+2)+0.0001) * self.mask)
            dets.append(log_det)
        dets=torch.Tensor(dets).cuda()
        logdet=logdet+dets
        return out_new,logdet
      else:
        out_new=input
        self.init_mask(permute)
        num_channels = self.weight.shape[0]
        # mask=torch.Tensor([[[1,0,0,1],[1,0,0,1]]]).permute(0,2,1)
        B, C, H, W = out_new.shape
        # print(mask)
        sig = torch.nn.Sigmoid()
        s_sig = sig(self.s+2)+0.0001
        s_sig_in = torch.ones_like(s_sig) / s_sig
        inp_masked = out_new.view(B, C,H * W) * self.mask * s_sig_in
        # print(inp_masked)
        out_conv = F.conv1d(inp_masked, self.weight, bias=self.bias)
        # print(out_conv)
        #pool1 = torch.nn.AvgPool1d(num_channels)
        pool_out = self.pool1(out_conv.transpose(1, 2))  # *2
        attn_out = (sig(pool_out.squeeze(2)+2)+0.0001).unsqueeze(1)
        # print(out_conv.transpose(1,2))
        attn_out = torch.ones_like(attn_out) / attn_out
        attn_mask = (1 - self.mask) * attn_out + self.mask * s_sig_in
        # print(attn_mask.shape)
        # print(attn_mask.view(B,C*H*W,T).permute(0,2,1).view(B,T,C,H,W))
        input_rev = out_new * (attn_mask.view(B, C,H * W).view(B, C, H, W))
        dets=[]
        for i in range(pool_out.shape[0]):
            log_det = 0
            for j in range(pool_out.shape[-2]):
                scale = pool_out[i, j, 0]
                log_det = log_det + (self.input_channels// 2) * torch.log(sig(scale+2)+0.0001)
            #print(log_det.shape)
            log_det = log_det + torch.sum(torch.log(sig(self.s+2)+0.0001) * self.mask)
            dets.append(log_det)
        dets=torch.Tensor(dets).cuda()
        logdet=logdet-dets
        return input_rev,logdet'''
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
class _Spatial_first_order_attn(nn.Module):
    def __init__(self, input_channels):
        super(_Spatial_first_order_attn, self).__init__()
        #self.num_channels=num_channels
        self.input_channels=input_channels
        self.weight = torch.empty([self.input_channels,self.input_channels,1]).cuda()
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight=torch.nn.Parameter(self.weight.type(torch.DoubleTensor)).cuda()
        self.bias = torch.empty([self.input_channels]).cuda()
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.bias =torch.nn.Parameter(self.bias.type(torch.DoubleTensor)).cuda()
        self.s = torch.nn.Parameter(torch.randn([1,self.input_channels,1]).type(torch.DoubleTensor)).cuda()
        self.pool1 = torch.nn.AvgPool1d(self.input_channels)
        self.offset= (torch.nn.Parameter(torch.ones([1])*8).type(torch.DoubleTensor)).cuda()
    def init_mask(self,permute=False):
        mask = torch.zeros((1, self.num_channels,self.input_channels)).double().cuda()
        mask[:, 1::2, ::2] = 1
        mask[:, ::2, 1::2] = 1
        mask = mask.permute(0, 2, 1)
        if permute:
            mask=1-mask
        self.mask=mask

    def forward(self, input: torch.Tensor, logdet=0, reverse=False, permute=False):
        # print(input)
        # print(weight)
      if not reverse:
        self.num_channels=input.shape[-1]**2
        self.init_mask(permute)
        input = input
        B, C, H, W = input.shape

        sig = torch.nn.Sigmoid()
        input_masked = input.view(B, C, H * W) * self.mask
        z = F.conv1d(input_masked, self.weight, bias=self.bias)
        z_new = z.transpose(1, 2)
        pool_out = self.pool1(z_new)
        #print(pool_out.shape)
        attn_out = (sig(pool_out.squeeze(-1)+self.offset)+0.000001).unsqueeze(1)
        #print(attn_out.shape)
        attn_mask = (1 - self.mask) * attn_out + self.mask * (sig(self.s)+0.000001)
        out_new = input * attn_mask.view(B, C,H * W).view(B, C, H, W)
        #log_det_final = 0
        #print(pool_out.shape)
        #dets=[]
        '''for i in range(pool_out.shape[0]):
            log_det = 0
            for j in range(pool_out.shape[-2]):
                scale = pool_out[i, j, 0]
                log_det = log_det + (self.input_channels// 2) * torch.log(sig(scale+self.offset)+0.0001)
            print(log_det)
        '''
        print(logdet)
        logdet = logdet + torch.sum((self.input_channels// 2) * (torch.log(sig(pool_out.squeeze(-1)+self.offset)+0.000001)),dim=-1)
        #print(logdet)
        logdet = logdet + torch.sum(torch.log(sig(self.s)+0.000001) * self.mask)
        #dets.append(log_det)
        #dets=torch.Tensor(dets)#.cuda()
        #logdet=logdet+dets
        return out_new,logdet
      else:
        out_new=input
        self.init_mask(permute)
        num_channels = self.weight.shape[0]
        # mask=torch.Tensor([[[1,0,0,1],[1,0,0,1]]]).permute(0,2,1)
        B, C, H, W = out_new.shape
        # print(mask)
        sig = torch.nn.Sigmoid()
        s_sig = sig(self.s)+0.000001
        s_sig_in = torch.ones_like(s_sig) / s_sig
        inp_masked = out_new.view(B, C,H * W) * self.mask * s_sig_in
        # print(inp_masked)
        out_conv = F.conv1d(inp_masked, self.weight, bias=self.bias)
        # print(out_conv)
        #pool1 = torch.nn.AvgPool1d(num_channels)
        pool_out = self.pool1(out_conv.transpose(1, 2))  # *2
        attn_out = (sig(pool_out.squeeze(2)+self.offset)+0.000001).unsqueeze(1)
        # print(out_conv.transpose(1,2))
        attn_out = torch.ones_like(attn_out) / attn_out
        attn_mask = (1 - self.mask) * attn_out + self.mask * s_sig_in
        # print(attn_mask.shape)
        # print(attn_mask.view(B,C*H*W,T).permute(0,2,1).view(B,T,C,H,W))
        input_rev = out_new * (attn_mask.view(B, C,H * W).view(B, C, H, W))
        logdet = logdet - torch.sum((self.input_channels// 2) * (torch.log(sig(pool_out.squeeze(-1)+self.offset)+0.000001)),dim=-1)
        logdet = logdet - torch.sum(torch.log(sig(self.s)+0.000001) * self.mask)
        return input_rev,logdet
#attn=Transformer_attn(3)
#inp=torch.randn([3,3,160,160],requires_grad=True)
#out,det=attn(inp,logdet=0, reverse=False)
#print(out)
#print(det)
#out_rev,det=attn(out,logdet=0, reverse=True)
#print(out_rev)
#print(inp)
#J=torch.autograd.functional.jacobian(attn, inp, create_graph=False, strict=False)
#print(torch.slogdet(J[0][0,:,:,:,0,:,:,:].view(160*160*3,160*160*3)))
'''class _Spatial_second_order_attn(nn.Module):
    def __init__(self,input_channels,num_frames):
        super(_Spatial_second_order_attn, self).__init__()
        self.input_channels=input_channels
        self.num_frames=num_frames
        self.weight1 = torch.empty([self.input_channels,self.input_channels,1])
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        self.weight1 = torch.nn.Parameter(self.weight1)
        self.bias1 = torch.empty([self.input_channels])
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias1, -bound, bound)
        self.bias1 =torch.nn.Parameter(self.bias1)
        self.weight2 = torch.empty([self.input_channels,self.input_channels,1])
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        self.weight2 = torch.nn.Parameter(self.weight2)
        self.bias2 = torch.empty([self.input_channels])
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias2, -bound, bound)
        self.bias2 =torch.nn.Parameter(self.bias2)

    def init_mask(self,permute=False):
        mask = torch.zeros((1, self.num_frames,self.input_channels))
        mask[:, 1::2, ::2] = 1
        mask[:, ::2, 1::2] = 1
        mask = mask.permute(0, 2, 1)
        if permute:
            mask=1-mask
        self.mask=mask

    def forward(self,input,permute=False):
        B, C, H, W = input.shape
        input = input.view(B, C, H * W)
        self.init_mask(permute)
        input_mask = input * self.mask
        sig = torch.nn.Sigmoid()
        out1 = torch.nn.functional.conv1d(input_mask, self.weight1,bias=self.bias1)
        out2 = torch.nn.functional.conv1d(input_mask, self.weight2,bias=self.bias2)
        attn = sig(torch.matmul(out1.permute(0, 2, 1), out2))
        out = torch.matmul(attn, input.permute(0, 2, 1)).view(B,C, H, W)
        mask = self.mask.permute(0, 2, 1).view(1, C, H, W)
        out_final = out * (1 - mask) + mask * input.view(B, C, H, W)
        log_det_final = 0
        attn_diag = torch.diagonal(attn, dim1=-2, dim2=-1)
        for i in range(attn_diag.shape[0]):
            log_det = 0
            for j in range(attn_diag.shape[-1]):
                scale = attn_diag[i, j]
                log_det = log_det + (C / 2) * torch.log(scale)
            log_det_final = log_det_final + log_det
        print(log_det_final)
        return out_final,log_det_final

    def reverse(self,out_final,permute=False):
        B, C, H, W = input.shape
        self.init_mask(permute)
        mask = self.mask.permute(0, 2, 1).view(1, C, H, W)
        input_masked_rev = out_final * mask
        sig = torch.nn.Sigmoid()
        input_masked_rev = input_masked_rev.view(B, C , H * W)
        out1_rev = torch.nn.functional.conv1d(input_masked_rev, self.weight1,bias=self.bias1)
        out2_rev = torch.nn.functional.conv1d(input_masked_rev, self.weight2,bias=self.bias2)
        attn_rev = torch.matmul(out1_rev.permute(0, 2, 1), out2_rev)  # .unsqueeze(2)
        # mask=1-mask
        #print(sig(attn_rev))
        outf_masked = (out_final.view(B, C ,H * W) * (1 - self.mask)).view(B,C,H,W)
        # print((torch.diagonal(attn_rev).unsqueeze(-1).repeat(1,1,C*H*W)).shape)
        v = (outf_masked - torch.matmul(sig(attn_rev), (input_masked_rev).permute(0, 2, 1)).view(B, H, W,C).permute(0,3,1,2)).view(B,H * W,C) * 1 / sig((torch.diagonal(attn_rev, dim1=-2, dim2=-1).unsqueeze(-1).repeat(1, 1, C)))
        v = v.permute(0,2,1).view(B, C, H, W)
        # print(mask.permute(0,2,1).view(1,T,C,H,W)*out_final)
        # print(v)
        # input_rev=((1-mask.permute(0,2,1).view(1,T,C,H,W))+mask.permute(0,2,1).view(1,T,C,H,W)*(-1))*v
        input_rev = self.mask.view(1,C,H,W)* out_final + (
                    1 - self.mask.view(1, C, H, W)) * v
        return input_rev

class _Temporal_second_order_attn(nn.Module):
    def __init__(self,input_channels,num_frames):
        super(_Temporal_second_order_attn, self).__init__()
        self.input_channels=input_channels
        self.num_frames=num_frames
        self.weight1 = torch.empty([self.input_channels,self.input_channels,1])
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        self.weight1 = torch.nn.Parameter(self.weight1)
        self.bias1 = torch.empty([self.input_channels])
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias1, -bound, bound)
        self.bias1 =torch.nn.Parameter(self.bias1)
        self.weight2 = torch.empty([self.input_channels,self.input_channels,1])
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        self.weight2 = torch.nn.Parameter(self.weight2)
        self.bias2 = torch.empty([self.input_channels])
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias2, -bound, bound)
        self.bias2 =torch.nn.Parameter(self.bias2)

    def init_mask(self,permute=False):
        mask = torch.zeros((1, self.num_frames,self.input_channels))
        mask[:, 1::2, ::2] = 1
        mask[:, ::2, 1::2] = 1
        mask = mask.permute(0, 2, 1)
        if permute:
            mask=1-mask
        self.mask=mask

    def forward(self,input,permute=False):
        B, T, C, H, W = input.shape
        input = input.permute(0, 2, 3, 4, 1).view(B, C * H * W, T)
        self.init_mask(permute)
        input_mask = input * self.mask
        sig = torch.nn.Sigmoid()
        out1 = torch.nn.functional.conv1d(input_mask, self.weight1,bias=self.bias1)
        out2 = torch.nn.functional.conv1d(input_mask, self.weight2,bias=self.bias2)
        attn = sig(torch.matmul(out1.permute(0, 2, 1), out2))
        out = torch.matmul(attn, input.permute(0, 2, 1)).view(B, T, C, H, W)
        mask = self.mask.permute(0, 2, 1).view(1, T, C, H, W)
        out_final = out * (1 - mask) + mask * input.permute(0, 2, 1).view(B, T, C, H, W)
        log_det_final = 0
        attn_diag = torch.diagonal(attn, dim1=-2, dim2=-1)
        for i in range(attn_diag.shape[0]):
            log_det = 0
            for j in range(attn_diag.shape[-1]):
                scale = attn_diag[i, j]
                log_det = log_det + (C * H * W / 2) * torch.log(scale)
            log_det_final = log_det_final + log_det
        print(log_det_final)
        return out_final,log_det_final

    def reverse(self,out_final,permute=False):
        B, T, C, H, W = input.shape
        self.init_mask(permute)
        mask = self.mask.permute(0, 2, 1).view(1, T, C, H, W)
        input_masked_rev = out_final * mask
        sig = torch.nn.Sigmoid()
        input_masked_rev = input_masked_rev.permute(0, 2, 3, 4, 1).view(B, C * H * W, T)
        out1_rev = torch.nn.functional.conv1d(input_masked_rev, self.weight1,bias=self.bias1)
        out2_rev = torch.nn.functional.conv1d(input_masked_rev, self.weight2,bias=self.bias2)
        attn_rev = torch.matmul(out1_rev.permute(0, 2, 1), out2_rev)  # .unsqueeze(2)
        # mask=1-mask
        #print(sig(attn_rev))
        outf_masked = (out_final.permute(0, 2, 3, 4, 1).view(B, C * H * W, T) * (1 - self.mask)).permute(0, 2, 1).view(B, T,
                                                                                                                  C, H,
                                                                                                                  W)
        # print((torch.diagonal(attn_rev).unsqueeze(-1).repeat(1,1,C*H*W)).shape)
        v = (outf_masked - torch.matmul(sig(attn_rev), (input_masked_rev).permute(0, 2, 1)).view(B, T, C, H, W)).view(B,
                                                                                                                      T,
                                                                                                                      C * H * W) * 1 / sig(
            (torch.diagonal(attn_rev, dim1=-2, dim2=-1).unsqueeze(-1).repeat(1, 1, C * H * W)))
        v = v.view(B, T, C, H, W)
        # print(mask.permute(0,2,1).view(1,T,C,H,W)*out_final)
        # print(v)
        # input_rev=((1-mask.permute(0,2,1).view(1,T,C,H,W))+mask.permute(0,2,1).view(1,T,C,H,W)*(-1))*v
        input_rev = self.mask.permute(0, 2, 1).view(1, T, C, H, W) * out_final + (
                    1 - self.mask.permute(0, 2, 1).view(1, T, C, H, W)) * v
        return input_rev

class _Channel_second_order_attn(nn.Module):
    def __init__(self,input_channels,num_frames):
        super(_Channel_second_order_attn, self).__init__()
        self.input_channels=input_channels
        self.num_frames=num_frames
        self.weight1 = torch.empty([self.input_channels,self.input_channels,1])
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        self.weight1 = torch.nn.Parameter(self.weight1)
        self.bias1 = torch.empty([self.input_channels])
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias1, -bound, bound)
        self.bias1 =torch.nn.Parameter(self.bias1)
        self.weight2 = torch.empty([self.input_channels,self.input_channels,1])
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        self.weight2 = torch.nn.Parameter(self.weight2)
        self.bias2 = torch.empty([self.input_channels])
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias2, -bound, bound)
        self.bias2 =torch.nn.Parameter(self.bias2)

    def init_mask(self,permute=False):
        mask = torch.zeros((1, self.num_frames,self.input_channels))
        mask[:, 1::2, ::2] = 1
        mask[:, ::2, 1::2] = 1
        mask = mask.permute(0, 2, 1)
        if permute:
            mask=1-mask
        self.mask=mask

    def forward(self,input,permute=False):
        B, C, H, W = input.shape
        input = input.permute(0, 2, 3, 1).view(B, H * W, C)
        self.init_mask(permute)
        input_mask = input * self.mask
        sig = torch.nn.Sigmoid()
        out1 = torch.nn.functional.conv1d(input_mask, self.weight1,bias=self.bias1)
        out2 = torch.nn.functional.conv1d(input_mask, self.weight2,bias=self.bias2)
        attn = sig(torch.matmul(out1.permute(0, 2, 1), out2))
        out = torch.matmul(attn, input.permute(0, 2, 1)).view(B, C, H, W)
        mask = self.mask.permute(0, 2, 1).view(1, C, H, W)
        out_final = out * (1 - mask) + mask * input.permute(0, 2, 1).view(B, C, H, W)
        log_det_final = 0
        attn_diag = torch.diagonal(attn, dim1=-2, dim2=-1)
        for i in range(attn_diag.shape[0]):
            log_det = 0
            for j in range(attn_diag.shape[-1]):
                scale = attn_diag[i, j]
                log_det = log_det + (H * W / 2) * torch.log(scale)
            log_det_final = log_det_final + log_det
        print(log_det_final)
        return out_final,log_det_final

    def reverse(self,out_final,permute=False):
        B, C, H, W = input.shape
        self.init_mask(permute)
        mask = self.mask.permute(0, 2, 1).view(1, C, H, W)
        input_masked_rev = out_final * mask
        sig = torch.nn.Sigmoid()
        input_masked_rev = input_masked_rev.permute(0, 2, 3, 1).view(B, H * W,C)
        out1_rev = torch.nn.functional.conv1d(input_masked_rev, self.weight1,bias=self.bias1)
        out2_rev = torch.nn.functional.conv1d(input_masked_rev, self.weight2,bias=self.bias2)
        attn_rev = torch.matmul(out1_rev.permute(0, 2, 1), out2_rev)  # .unsqueeze(2)
        # mask=1-mask
        #print(sig(attn_rev))
        outf_masked = (out_final.permute(0, 2, 3, 1).view(B, H * W,C) * (1 - self.mask)).permute(0, 2, 1).view(B,C, H,W)
        # print((torch.diagonal(attn_rev).unsqueeze(-1).repeat(1,1,C*H*W)).shape)
        v = (outf_masked - torch.matmul(sig(attn_rev), (input_masked_rev).permute(0, 2, 1)).view(B, C, H, W)).view(B, C, H * W) * 1 / sig(
            (torch.diagonal(attn_rev, dim1=-2, dim2=-1).unsqueeze(-1).repeat(1, 1, H * W)))
        v = v.view(B, C, H, W)
        # print(mask.permute(0,2,1).view(1,T,C,H,W)*out_final)
        # print(v)
        # input_rev=((1-mask.permute(0,2,1).view(1,T,C,H,W))+mask.permute(0,2,1).view(1,T,C,H,W)*(-1))*v
        input_rev = self.mask.permute(0, 2, 1).view(1, C, H, W) * out_final + (
                    1 - self.mask.permute(0, 2, 1).view(1, C, H, W)) * v
        return input_rev



class _Temporal_first_order_attn(nn.Module):
    def __init__(self, input_channels,num_frames):
        super(_Temporal_first_order_attn, self).__init__()
        self.num_frames=num_frames
        self.input_channels=input_channels
        self.weight = torch.empty([self.input_channels,self.input_channels,1])
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight=torch.nn.Parameter(self.weight)
        self.bias = torch.empty([self.input_channels])
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.bias =torch.nn.Parameter(self.bias)
        self.s = torch.nn.Parameter(torch.randn([1,self.input_channels,num_frames]))
        self.pool1 = torch.nn.AvgPool1d(self.input_channels)

    def init_mask(self,permute=False):
        mask = torch.zeros((1, self.num_frames,self.input_channels))
        mask[:, 1::2, ::2] = 1
        mask[:, ::2, 1::2] = 1
        mask = mask.permute(0, 2, 1)
        if permute:
            mask=1-mask
        self.mask=mask

    def forward(self,input,permute=False):
        # print(input)
        # print(weight)
        self.init_mask(permute)
        input = input
        B, T, C, H, W = input.shape
        sig = torch.nn.Sigmoid()
        input_masked = input.permute(0, 2, 3, 4, 1).view(B, C * H * W, T) * self.mask
        z = F.conv1d(input_masked, self.weight, bias=self.bias)
        z_new = z.transpose(1, 2)
        pool_out = self.pool1(z_new)
        attn_out = torch.exp(pool_out.squeeze(2)).unsqueeze(1)
        attn_mask = (1 - self.mask) * attn_out + self.mask * sig(self.s)
        out_new = input * attn_mask.view(B, C * H * W, T).permute(0, 2, 1).view(B, T, C, H, W)
        log_det_final = 0
        for i in range(pool_out.shape[0]):
            log_det = 0
            for j in range(pool_out.shape[-2]):
                scale = pool_out[i, j, 0]
                log_det = log_det + (C * H * W / 2) * scale
            log_det = log_det + torch.sum(torch.log(sig(self.s)) * self.mask)
            log_det_final = log_det_final + log_det
        return out_new,log_det_final

    def reverse(self,out_new,permute=False):
        self.init_mask(permute)
        num_channels = self.weight.shape[0]
        # mask=torch.Tensor([[[1,0,0,1],[1,0,0,1]]]).permute(0,2,1)
        B, T, C, H, W = out_new.shape
        # print(mask)
        sig = torch.nn.Sigmoid()
        s_sig = sig(self.s)
        s_sig_in = torch.ones_like(s_sig) / s_sig
        inp_masked = out_new.permute(0, 2, 3, 4, 1).view(B, C * H * W, T) * self.mask * s_sig_in
        # print(inp_masked)
        out_conv = F.conv1d(inp_masked, self.weight, bias=self.bias)
        # print(out_conv)
        pool1 = torch.nn.AvgPool1d(num_channels)
        pool_out = pool1(out_conv.transpose(1, 2))  # *2
        pool_out = torch.exp(pool_out.squeeze(2)).unsqueeze(1)
        # print(out_conv.transpose(1,2))
        pool_out = torch.ones_like(pool_out) / pool_out
        attn_mask = (1 - self.mask) * pool_out + self.mask * s_sig_in
        # print(attn_mask.shape)
        # print(attn_mask.view(B,C*H*W,T).permute(0,2,1).view(B,T,C,H,W))
        input_rev = out_new * (attn_mask.view(B, C * H * W, T).permute(0, 2, 1).view(B, T, C, H, W))
        return input_rev

class _Channel_first_order_attn(nn.Module):
    def __init__(self, input_channels,num_channels):
        super(_Channel_first_order_attn, self).__init__()
        self.num_channels=num_channels
        self.input_channels=input_channels
        self.weight = torch.empty([self.input_channels,self.input_channels,1])
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight=torch.nn.Parameter(self.weight)
        self.bias = torch.empty([self.input_channels])
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.bias =torch.nn.Parameter(self.bias)
        self.s = torch.nn.Parameter(torch.randn([1,self.input_channels,num_channels]))
        self.pool1 = torch.nn.AvgPool1d(self.input_channels)

    def init_mask(self,permute=False):
        mask = torch.zeros((1, self.num_channels,self.input_channels))
        mask[:, 1::2, ::2] = 1
        mask[:, ::2, 1::2] = 1
        mask = mask.permute(0, 2, 1)
        if permute:
            mask=1-mask
        self.mask=mask

    def forward(self,input,permute=False):
        # print(input)
        # print(weight)
        self.init_mask(permute)
        input = input
        B, C, H, W = input.shape
        sig = torch.nn.Sigmoid()
        input_masked = input.permute(0, 2, 3, 1).view(B, H * W, C) * self.mask
        z = F.conv1d(input_masked, self.weight, bias=self.bias)
        z_new = z.transpose(1, 2)
        pool_out = self.pool1(z_new)
        attn_out = torch.exp(pool_out.squeeze(2)).unsqueeze(1)
        attn_mask = (1 - self.mask) * attn_out + self.mask * sig(self.s)
        out_new = input * attn_mask.view(B, H * W, C).permute(0, 2, 1).view(B, C, H, W)
        log_det_final = 0
        dets = []
        for i in range(pool_out.shape[0]):
            log_det = 0
            for j in range(pool_out.shape[-2]):
                scale = pool_out[i, j, 0]
                log_det = log_det + (self.input_channels/ 2) * scale
            log_det = log_det + torch.sum(torch.log(sig(self.s)) * self.mask)
            dets.append(log_det)
        dets = torch.Tensor(dets)
        return out_new,dets

    def reverse(self,out_new,permute=False):
        self.init_mask(permute)
        num_channels = self.weight.shape[0]
        # mask=torch.Tensor([[[1,0,0,1],[1,0,0,1]]]).permute(0,2,1)
        B, C, H, W = out_new.shape
        # print(mask)
        sig = torch.nn.Sigmoid()
        s_sig = sig(self.s)
        s_sig_in = torch.ones_like(s_sig) / s_sig
        inp_masked = out_new.permute(0, 2, 3, 1).view(B, H * W, C) * self.mask * s_sig_in
        # print(inp_masked)
        out_conv = F.conv1d(inp_masked, self.weight, bias=self.bias)
        # print(out_conv)
        pool1 = torch.nn.AvgPool1d(num_channels)
        pool_out = pool1(out_conv.transpose(1, 2))  # *2
        pool_out = torch.exp(pool_out.squeeze(2)).unsqueeze(1)
        # print(out_conv.transpose(1,2))
        pool_out = torch.ones_like(pool_out) / pool_out
        attn_mask = (1 - self.mask) * pool_out + self.mask * s_sig_in
        # print(attn_mask.shape)
        # print(attn_mask.view(B,C*H*W,T).permute(0,2,1).view(B,T,C,H,W))
        input_rev = out_new * (attn_mask.view(B, H * W, C).permute(0, 2, 1).view(B, C, H, W))
        return input_rev

class _Spatial_first_order_attn(nn.Module):
    def __init__(self, input_channels,num_channels):
        super(_Spatial_first_order_attn, self).__init__()
        self.num_channels=num_channels
        self.input_channels=input_channels
        self.weight = torch.empty([self.input_channels,self.input_channels,1])
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight=torch.nn.Parameter(self.weight)
        self.bias = torch.empty([self.input_channels])
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.bias =torch.nn.Parameter(self.bias)
        self.s = torch.nn.Parameter(torch.randn([1,self.input_channels,num_channels]))
        self.pool1 = torch.nn.AvgPool1d(self.input_channels)

    def init_mask(self,permute=False):
        mask = torch.zeros((1, self.num_channels,self.input_channels))
        mask[:, 1::2, ::2] = 1
        mask[:, ::2, 1::2] = 1
        mask = mask.permute(0, 2, 1)
        if permute:
            mask=1-mask
        self.mask=mask

    def forward(self,input,permute=False):
        # print(input)
        # print(weight)
        self.init_mask(permute)
        input = input
        B, C, H, W = input.shape
        sig = torch.nn.Sigmoid()
        input_masked = input.view(B, C, H * W) * self.mask
        z = F.conv1d(input_masked, self.weight, bias=self.bias)
        z_new = z.transpose(1, 2)
        pool_out = self.pool1(z_new)
        attn_out = torch.exp(pool_out.squeeze(2)).unsqueeze(1)
        attn_mask = (1 - self.mask) * attn_out + self.mask * sig(self.s)
        out_new = input * attn_mask.view(B, C,H * W).view(B, C, H, W)
        log_det_final = 0
        print(pool_out.shape[0])
        dets=[]
        for i in range(pool_out.shape[0]):
            log_det = 0
            for j in range(pool_out.shape[-2]):
                scale = pool_out[i, j, 0]
                log_det = log_det + (self.input_channels/ 2) * scale
            #print(log_det.shape)
            log_det = log_det + torch.sum(torch.log(sig(self.s)) * self.mask)
            dets.append(log_det)
        dets=torch.Tensor(dets)
        return out_new,dets

    def reverse(self,out_new,permute=False):
        self.init_mask(permute)
        num_channels = self.weight.shape[0]
        # mask=torch.Tensor([[[1,0,0,1],[1,0,0,1]]]).permute(0,2,1)
        B, C, H, W = out_new.shape
        # print(mask)
        sig = torch.nn.Sigmoid()
        s_sig = sig(self.s)
        s_sig_in = torch.ones_like(s_sig) / s_sig
        inp_masked = out_new.view(B, C,H * W) * self.mask * s_sig_in
        # print(inp_masked)
        out_conv = F.conv1d(inp_masked, self.weight, bias=self.bias)
        # print(out_conv)
        pool1 = torch.nn.AvgPool1d(num_channels)
        pool_out = pool1(out_conv.transpose(1, 2))  # *2
        pool_out = torch.exp(pool_out.squeeze(2)).unsqueeze(1)
        # print(out_conv.transpose(1,2))
        pool_out = torch.ones_like(pool_out) / pool_out
        attn_mask = (1 - self.mask) * pool_out + self.mask * s_sig_in
        # print(attn_mask.shape)
        # print(attn_mask.view(B,C*H*W,T).permute(0,2,1).view(B,T,C,H,W))
        input_rev = out_new * (attn_mask.view(B, C,H * W).view(B, C, H, W))
        return input_rev
input=torch.nn.Parameter(torch.randn([2,16,8,8]))
input_channels=16
num_frames=8*8
attn=_Spatial_first_order_attn(input_channels,num_frames)
out,logdet=attn(input)
inp_rev=attn.reverse(out)
print(inp_rev)
print(input)
print(logdet)
a=torch.autograd.functional.jacobian(attn, (input))
j0= a[0][0,:,:,:,0,:,:,:].view(num_frames*input_channels,num_frames*input_channels)
j1=a[0][1,:,:,:,1,:,:,:].view(num_frames*input_channels,num_frames*input_channels)
print(torch.logdet(j0)+torch.logdet(j1))
########################'''
'''input=torch.nn.Parameter(torch.randn([2,4*2,8,8]))
input_channels=8*8
num_frames=16
attn=_Channel_first_order_attn(input_channels,num_frames)
out,logdet=attn(input)
inp_rev=attn.reverse(out)
print(inp_rev)
print(input)
print(logdet)
a=torch.autograd.functional.jacobian(attn, (input))
j0= a[0][0,:,:,:,0,:,:,:].view(16*8*8,16*8*8)
j1=a[0][1,:,:,:,1,:,:,:].view(16*8*8,16*8*8)
print(torch.logdet(j0)+torch.logdet(j1))

input=torch.nn.Parameter(torch.randn([2,2,2,8,8]))
input_channels=8*8*2
num_frames=2
attn=_Temporal_second_order_attn(input_channels,num_frames)
out,logdet=attn(input)
inp_rev=attn.reverse(out)
print(inp_rev)
print(input)
a=torch.autograd.functional.jacobian(attn, (input))
j0= a[0][0,:,:,:,:,0,:,:,:,:].view(2*2*8*8,4*8*8)
j1=a[0][1,:,:,:,:,1,:,:,:,:].view(4*8*8,4*8*8)
print(torch.logdet(j0)+torch.logdet(j1))

input=torch.nn.Parameter(torch.randn([2,2,8,8]))
input_channels=8*8
num_frames=2
attn=_Channel_second_order_attn(input_channels,num_frames)
out,logdet=attn(input)
inp_rev=attn.reverse(out)
print(inp_rev)
print(input)
a=torch.autograd.functional.jacobian(attn, (input))
j0= a[0][0,:,:,:,0,:,:,:].view(2*8*8,2*8*8)
j1=a[0][1,:,:,:,1,:,:,:].view(2*8*8,2*8*8)
print(torch.logdet(j0)+torch.logdet(j1))

input=torch.nn.Parameter(torch.randn([2,2,8,8]))
input_channels=2
num_frames=8*8
attn=_Spatial_second_order_attn(input_channels,num_frames)
out,logdet=attn(input)
inp_rev=attn.reverse(out)
print(inp_rev)
print(input)
a=torch.autograd.functional.jacobian(attn, (input))
j0= a[0][0,:,:,:,0,:,:,:].view(2*8*8,2*8*8)
j1=a[0][1,:,:,:,1,:,:,:].view(2*8*8,2*8*8)
print(torch.logdet(j0)+torch.logdet(j1))'''
