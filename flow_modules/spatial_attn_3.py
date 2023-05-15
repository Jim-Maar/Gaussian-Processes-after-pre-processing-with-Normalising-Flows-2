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
        self.weight=torch.nn.Parameter(self.weight).cuda()
        self.bias = torch.empty([self.input_channels]).cuda()
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.bias =torch.nn.Parameter(self.bias).cuda()
        self.register_parameter("s",nn.Parameter(torch.randn([1,self.input_channels,1])))
        self.register_parameter("offset",nn.Parameter(torch.ones([1])*8))
        #self.s = torch.nn.Parameter(torch.randn([1,self.input_channels,1])).double()
        self.pool1 = torch.nn.AvgPool1d(self.input_channels)
        #self.offset= (torch.nn.Parameter(torch.ones([1])*8)).double()
    def init_mask(self,permute=False):
        mask = torch.zeros((1, self.num_channels,self.input_channels)).cuda()
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
        #print(logdet)
        logdet = logdet + torch.sum((self.input_channels// 2) * (torch.log(sig(pool_out.squeeze(-1)+self.offset)+0.000001)),dim=-1)
        #print(logdet)
        logdet = logdet + torch.sum(torch.log(sig(self.s)+0.000001) * self.mask)
        #dets.append(log_det)
        #dets=torch.Tensor(dets)#.cuda()
        #logdet=logdet+dets
        return out_new,logdet
      else:
        out_new=input
        #self.init_mask(permute)
        num_channels=self.num_channels = input.shape[-1]**2
        self.init_mask(permute)
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
