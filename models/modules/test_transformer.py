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
        self.convq = torch.empty([num_channels,num_channels,1,1]).type(torch.DoubleTensor).cuda()
        nn.init.kaiming_uniform_(self.convq, a=math.sqrt(5))
        self.convk = torch.empty([num_channels,num_channels,1,1]).type(torch.DoubleTensor).cuda()
        nn.init.kaiming_uniform_(self.convk, a=math.sqrt(5))
        #self.convq=torch.nn.Parameter(self.convq.type(torch.DoubleTensor))#.cuda()
        #self.convk =  torch.nn.Parameter(torch.Tensor(w_init2)).double()#.cuda()#InvertibleConv1x1(self.c).double()
        self.offset= (torch.nn.Parameter(torch.ones([1])*(0.99)).type(torch.DoubleTensor)).cuda()
        self.offset2= (torch.nn.Parameter(torch.ones([1])*(8)).type(torch.DoubleTensor)).cuda()
        self.offset3= (torch.nn.Parameter(torch.ones([1])*(0.001)).type(torch.DoubleTensor)).cuda()
        
    def forward(self, input: torch.Tensor, logdet=0, reverse=False, ft=None):
     if not reverse:
        #logdet=0
        #input=z
        z=input
        p =z.shape[-1]//2
        #print("Patch size",p)
        full_inp = rearrange(z, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        mask = torch.Tensor(checkerboard(full_inp[0, :, :].shape)).type(torch.DoubleTensor).cuda()
        f_m = full_inp * mask
        num_patches_sqrt = z.shape[-1] // p
        z_m = reverse_rearrange(f_m, p, num_patches_sqrt ** 2, z.shape)
        q= torch.nn.functional.conv2d(z_m,self.convq)
        k= torch.nn.functional.conv2d(z_m,self.convk)
        s = nn.LeakyReLU(0.0001)
        full_inp_q = (rearrange(q, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        full_inp_k = (rearrange(k, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        s = nn.LeakyReLU(0.0001) #torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        attn_mask = torch.Tensor(checkerboard([full_inp_q.shape[1],full_inp_q.shape[1]])).type(torch.DoubleTensor).cuda()
        attn = (s(torch.matmul(full_inp_q, full_inp_k.permute(0, 2, 1))+self.offset2)+self.offset3)*attn_mask
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

        print("Attn logdet",logdet)
        out = torch.matmul(attn, full_inp * (1 - mask))
        out_final = out * (1 - mask) + full_inp * mask
        z_out = reverse_rearrange(out_final, p, num_patches_sqrt ** 2, z.shape)
        output=z_out

     else:
        z=input
        p = z.shape[-1]//2
        out_final = rearrange(z, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        num_patches_sqrt = z.shape[-1] // p
        s = nn.LeakyReLU(0.0001) # torch.nn.Sigmoid()
        mask = torch.Tensor(checkerboard(out_final[0, :, :].shape)).type(torch.DoubleTensor).cuda()
        rev = out_final * (mask)
        rev_rearrange = reverse_rearrange(rev, p, num_patches_sqrt ** 2, z.shape)
        q= torch.nn.functional.conv2d(rev_rearrange,self.convq)
        k= torch.nn.functional.conv2d(rev_rearrange,self.convk)
        full_inp_q_rev = (rearrange(q, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        full_inp_k_rev = (rearrange(k, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p))
        s = nn.LeakyReLU(0.0001) #torch.nn.Sigmoid()
        #f=torch.eye(m1.shape[-1])*(1.001)
        attn_mask = torch.Tensor(checkerboard([full_inp_q_rev.shape[1], full_inp_q_rev.shape[1]])).type(torch.DoubleTensor).cuda()
        attn = (s(torch.matmul(full_inp_q_rev, full_inp_k_rev.permute(0, 2, 1))+self.offset2)+self.offset3)*attn_mask
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
        mask2 = torch.Tensor(checkerboard([rev_unmask.shape[1]])).cuda()
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
'''import torch
#x=torch.randn([1,3,8,8],requires_grad=True)
import torch.nn as nn
import numpy as np
import torch



 

attn=Transformer_attn(12)
s=torch.nn.Sigmoid()
inp=s(torch.randn([2,12,80,80],requires_grad=True)).double().cuda()
out,det=attn(inp,logdet=0, reverse=False)
#print(out)
print("Forward det",det)

out_rev,det=attn(out,logdet=0, reverse=True)
#print(out_rev)
print("Reverse det",det)
print(torch.sum(inp-out_rev))
#J=torch.autograd.functional.jacobian(attn, inp, create_graph=False, strict=False)
#print("Actual logdet 1",torch.slogdet(J[0][0,:,:,:,0,:,:,:].view(16*16*3,16*16*3))[1])
#print("Actual logdet 2",torch.slogdet(J[0][1,:,:,:,1,:,:,:].view(16*16*3,16*16*3))[1])
#print(torch.sum(inp-out_rev))'''
