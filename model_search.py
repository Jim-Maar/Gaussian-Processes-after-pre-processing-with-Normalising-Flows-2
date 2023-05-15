import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES, PRIMITIVES_attention
# from utils.darts_utils import drop_path, compute_speed, compute_speed_tensorrt
from pdb import set_trace as bp
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from util_gan.vgg_feature import VGGFeature
from thop import profile
from basicsr.models.archs.arch_util import (DCNv2Pack, ResidualBlockNoBN,
                                                    make_layer)
from thop import clever_format
ENABLE_TANH = False
from basicsr.models.archs.arch_util import DCNv2Pack as DCN
#try:
#    from models.archs.dcn.deform_conv import ModulatedDeformConvPack as DCN
#except ImportError:
#    raise ImportError('Failed to import DCNv2 module.')

try:
    from AIM2020_submit.codes.models.archs.deformable_kernels.modules import GlobalDeformKernel2d as GDK
    from AIM2020_submit.codes.models.archs.deformable_kernels.modules import DeformKernelConv2d as DKC
except ImportError:
    raise ImportError('Failed to import Deformable kernels')

# https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class SingleOp(nn.Module):
    def __init__(self, op, C_in, C_out, kernel_size=3, stride=1, slimmable=True, width_mult_list=[1.], quantize=True):
        super(SingleOp, self).__init__()
        self._op = op(C_in, C_out, kernel_size=kernel_size, stride=stride, slimmable=slimmable,
                      width_mult_list=width_mult_list)
        self._width_mult_list = width_mult_list
        self.quantize = quantize
        self.slimmable = slimmable

    def set_prun_ratio(self, ratio):
        self._op.set_ratio(ratio)

    def forward(self, x, beta, ratio):
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        if self.quantize == 'search':
            result = (beta[0] * self._op(x, quantize=False) + beta[1] * self._op(x,
                                                                                 quantize=True)) * r_score0 * r_score1
        elif self.quantize:
            result = self._op(x, quantize=True) * r_score0 * r_score1
        else:
            result = self._op(x, quantize=False) * r_score0 * r_score1

        return result

    def forward_flops(self, size, beta, ratio):
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        if self.quantize == 'search':
            flops_full, size_out = self._op.forward_flops(size, quantize=False)
            flops_quant, _ = op.forward_flops(size, quantize=True)
            flops = beta[0] * flops_full + beta[1] * flops_quant
        elif self.quantize:
            flops, size_out = op.forward_flops(size, quantize=True)
        else:
            flops, size_out = self._op.forward_flops(size, quantize=False)

        flops = flops * r_score0 * r_score1

        return flops, size_out


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, stride=1, slimmable=True, width_mult_list=[1.], quantize=True):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._width_mult_list = width_mult_list
        self.quantize = quantize
        self.slimmable = slimmable

        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out, stride, slimmable=slimmable, width_mult_list=width_mult_list)
            self._ops.append(op)

    def set_prun_ratio(self, ratio):
        for op in self._ops:
            op.set_ratio(ratio)

    def forward(self, x, alpha, beta, ratio):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        for w, op in zip(alpha, self._ops):
            #print
            if self.quantize == 'search':
                result = result + (
                            beta[0] * op(x, quantize=False) + beta[1] * op(x, quantize=True)) * w * r_score0 * r_score1
            elif self.quantize:
                result = result + op(x, quantize=True) * w * r_score0 * r_score1
            else:
                result = result + op(x, quantize=False) * w * r_score0 * r_score1
            # print(type(op), result.shape)
        return result

    def forward_latency(self, size, alpha, ratio):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        for w, op in zip(alpha, self._ops):
            latency, size_out = op.forward_latency(size)
            result = result + latency * w * r_score0 * r_score1
        return result, size_out

    def forward_flops(self, size, alpha, beta, ratio):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        for w, op in zip(alpha, self._ops):
            if self.quantize == 'search':
                flops_full, size_out = op.forward_flops(size, quantize=False)
                flops_quant, _ = op.forward_flops(size, quantize=True)
                flops = (beta[0] * flops_full + beta[1] * flops_quant)

            elif self.quantize:
                flops, size_out = op.forward_flops(size, quantize=True)

            else:
                flops, size_out = op.forward_flops(size, quantize=False)

            result = result + flops * w * r_score0 * r_score1

        return result, size_out
class MixedOp_attn(nn.Module):

    def __init__(self,num_frames,num_channels_in):
        super(MixedOp_attn, self).__init__()
        self._ops = nn.ModuleList()

        for primitive in PRIMITIVES_attention:
            op = OPS_Attention[primitive](num_channels_in,num_frames)
            print(op)
            self._ops.append(op)

        self._ops_level2 = nn.ModuleList()

        for primitive in PRIMITIVES_attention:
            op = OPS_Attention[primitive](num_channels_in,num_frames)
            self._ops_level2.append(op)

    def forward(self, x, alpha_levels,alpha_sink,alpha_acts):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        multi_level=True
        '''for w, op in zip(alpha, self._ops):
            #print
            result = result + op(x) * w
            print(result.shape)
            # print(type(op), result.shape)'''
        alpha_acts_l1=alpha_acts[0:len(PRIMITIVES_attention),:]
        alpha_acts_l2=alpha_acts[len(PRIMITIVES_attention):,:]
        node_num_second=0
        if multi_level==True:
           res_final=0
           for i,op2 in enumerate(self._ops_level2):
               node_num_first = 0
               weights=alpha_levels[i,:]
               res=0
               for j, op in enumerate(self._ops):
                   res=res+op(x,alpha_acts_l1[node_num_first,:])*weights[j]
                   node_num_first=node_num_first+1
               res_final=res_final+op2(res,alpha_acts_l2[node_num_second,:])*alpha_sink[i]
               node_num_second=node_num_second+1
        node_num_first=0
        start_sink2=len(self._ops_level2)
        for k,op in enumerate(self._ops):
            res_final=res_final+alpha_sink[k+start_sink2]*op(x,alpha_acts_l1[node_num_first,:])
            node_num_first=node_num_first+1
        result=res_final
        return result

    def forward_latency(self, size, alpha, ratio):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        for w, op in zip(alpha, self._ops):
            latency, size_out = op.forward_latency(size)
            result = result + latency * w
        return result, size_out

    def forward_flops(self, size, alpha, beta, ratio):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        for w, op in zip(alpha, self._ops):
            flops, size_out = op.forward_flops(size)

            result = result + flops * w

        return result, size_out

class Cell(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf=64, op_per_cell=5, slimmable=True, width_mult_list=[1.], quantize=False):
        super(Cell, self).__init__()

        self.nf = nf
        self.op_per_cell = op_per_cell
        self.slimmable = slimmable
        self._width_mult_list = width_mult_list
        self.quantize = quantize

        self.ops = nn.ModuleList()

        for _ in range(op_per_cell):
            self.ops.append(
                MixedOp(self.nf, self.nf, slimmable=slimmable, width_mult_list=width_mult_list, quantize=quantize))

    def forward(self, x, alpha, beta, ratio):
        out = x
        #print("Staring loop")
        for i, op in enumerate(self.ops):
            #print(op)
            if i == 0:
                out = op(out, alpha[i], beta[i], [1, ratio[i]])
            elif i == self.op_per_cell - 1:
                out = op(out, alpha[i], beta[i], [ratio[i - 1], 1])
            else:
                out = op(out, alpha[i], beta[i], [ratio[i - 1], ratio[i]])
            #print(out.shape)
        return out * 0.2 + x

    def forward_flops(self, size, alpha, beta, ratio):
        flops_total = []

        for i, op in enumerate(self.ops):
            if i == 0:
                flops, size = op.forward_flops(size, alpha[i], beta[i], [1, ratio[i]])
                flops_total.append(flops)
            elif i == self.op_per_cell - 1:
                flops, size = op.forward_flops(size, alpha[i], beta[i], [ratio[i - 1], 1])
                flops_total.append(flops)
            else:
                flops, size = op.forward_flops(size, alpha[i], beta[i], [ratio[i - 1], ratio[i]])
                flops_total.append(flops)

        return sum(flops_total), size
class Cell_attn(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, num_frames,num_channels_in,num_channels_in_2,size,op_per_cell):
        super(Cell_attn, self).__init__()

        self.ops = nn.ModuleList()

        for _ in range(op_per_cell):
            self.ops.append(
                MixedOp_attn(num_frames,num_channels_in))

    def forward(self, x, alpha_levels,alpha_sink,alpha_acts):
        out = x
        #print("Staring loop")
        for i, op in enumerate(self.ops):
            #print(op)
            if i == 0:
                out = op(out, alpha_levels[i],alpha_sink[i],alpha_acts[i])
            elif i == self.op_per_cell - 1:
                out = op(out, alpha_levels[i],alpha_sink[i],alpha_acts[i])
            else:
                out = op(out, alpha_levels[i],alpha_sink[i],alpha_acts[i])
            #print(out.shape)
        return out * 0.2 + x

    def forward_flops(self, size, alpha):
        flops_total = []

        for i, op in enumerate(self.ops):
            if i == 0:
                flops, size = op.forward_flops(size, alpha[i])
                flops_total.append(flops)
            elif i == self.op_per_cell - 1:
                flops, size = op.forward_flops(size, alpha[i])
                flops_total.append(flops)
            else:
                flops, size = op.forward_flops(size, alpha[i])
                flops_total.append(flops)

        return sum(flops_total), size

class NAS_GAN(nn.Module):
    def __init__(self,alignment_type,fusion_type, num_cell=2, op_per_cell=5, slimmable=True, width_mult_list=[1., ],
                 loss_weight=[1e0, 1e5, 1e0, 1e-7],
                 prun_modes='arch_ratio', loss_func='MSE', before_act=True, quantize=False):

        super(NAS_GAN, self).__init__()
        self.align_type= alignment_type
        self.fusion_type=fusion_type
        self.num_cell =5#num_cell
        self.num_cell_attn=1
        self.op_per_cell_attn=1
        self.op_per_cell = 5#op_per_cell

        self._layers = self.num_cell * self.op_per_cell

        self._width_mult_list = width_mult_list
        self._prun_modes = prun_modes
        self.prun_mode = None  # prun_mode is higher priority than _prun_modes
        self._flops = 0
        self._params = 0
        self.center_HR=1
        self.base_weight = loss_weight[0]
        self.style_weight = loss_weight[1]
        self.content_weight = loss_weight[2]
        self.tv_weight = loss_weight[3]
        self.center=4
        self.shift_loss=True
        self.video_loss=True
        self.use_mean_var=True
        self.vgg = torch.nn.DataParallel(VGGFeature(before_act=before_act)).cuda()

        self.quantize = quantize
        self.slimmable = slimmable
        self.H=32
        self.W=32
        self.nf = 64
        self.num_frames=9
        #self.alpha=2
        #self.beta=3
        self.conv_first = Conv(3, self.nf, 3, 1, 1, bias=True)

        if self.align_type=="PCD":
           self.align = PCD_Align(num_feat=self.nf)
           self.fea_L2_conv1 = Conv(self.nf, self.nf, 3, 2, 1, bias=True)
           self.fea_L2_conv2 = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        elif self.align_type=="DKC":
           self.align = Align_fea(nf=self.nf, groups=8)#.cuda()
        self.cells_attn=nn.ModuleList()
        self.cells_level2=64
        for i in range(self.num_cell_attn):
            cell = Cell_attn(self.num_frames,self.cells_level2,self.cells_level2,8,self.op_per_cell_attn)
            self.cells_attn.append(cell)
        self.cells_pre = nn.ModuleList()
        for i in range(self.num_cell):
            cell = Cell(self.nf, op_per_cell=self.op_per_cell, slimmable=slimmable, width_mult_list=width_mult_list,
                        quantize=quantize)
            self.cells_pre.append(cell)
        self.cells_recon = nn.ModuleList()
        for i in range(self.num_cell):
            cell = Cell(self.nf, op_per_cell=self.op_per_cell, slimmable=slimmable, width_mult_list=width_mult_list,
                        quantize=quantize)
            self.cells_recon.append(cell)
        self.conv_attn_final=Conv(self.nf* self.num_frames, self.nf, 3, 1, 1,bias=True)
        self.conv11=Conv(self.nf,self.nf//2,3,1,1,bias=True)
        self.conv112=Conv(self.nf//2,3*2*2,3,1,1,bias=True)
        self.conv12=Conv(3,3,3,1,1,bias=True)
        self.conv122=Conv(3,3*2*2,3,1,1,bias=True)
        self.conv13 = Conv(3,3, 3, 1, 1, bias=True)
        self.conv132 =Conv(3,3*2*2,3,1,1,bias=True)
        self.conv14= Conv(3,3,3,1,1,bias=True)
        self.conv142=Conv(3,3*2*2,3,1,1,bias=True)
        #frame_al=torch.nn.functional.interpolate(frame_al, size=[8,8], mode="bilinear")
        self.pixshuff=nn.PixelShuffle(2)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.tanh = nn.Tanh()

        self.loss_func = nn.MSELoss() if loss_func == 'MSE' else nn.L1Loss()

        self._arch_params = self._build_arch_parameters()
        self._reset_arch_parameters()

    def sample_prun_ratio(self, mode="arch_ratio"):
        '''
        mode: "min"|"max"|"random"|"arch_ratio"(default)
        '''
        assert mode in ["min", "max", "random", "arch_ratio"]
        if mode == "arch_ratio":
            ratio = self._arch_params["ratio"]
            ratio_sampled = []
            for cell_id in range(self.num_cell):
                ratio_cell = []
                for op_id in range(self.op_per_cell - 1):
                    ratio_cell.append(gumbel_softmax(F.log_softmax(ratio[cell_id][op_id], dim=-1), hard=True))
                ratio_sampled.append(ratio_cell)

            return ratio_sampled

        elif mode == "min":
            ratio_sampled = []
            for cell_id in range(self.num_cell):
                ratio_cell = []
                for op_id in range(self.op_per_cell - 1):
                    ratio_cell.append(self._width_mult_list[0])
                ratio_sampled.append(ratio_cell)

            return ratio_sampled

        elif mode == "max":
            ratio_sampled = []
            for cell_id in range(self.num_cell):
                ratio_cell = []
                for op_id in range(self.op_per_cell - 1):
                    ratio_cell.append(self._width_mult_list[-1])
                ratio_sampled.append(ratio_cell)

            return ratio_sampled

        elif mode == "random":
            ratio_sampled = []

            for cell_id in range(self.num_cell):
                ratio_cell = []
                for op_id in range(self.op_per_cell - 1):
                    ratio_cell.append(np.random.choice(self._width_mult_list))
                ratio_sampled.append(ratio_cell)

            return ratio_sampled

    def forward(self, input): 
     #output=[]
     #input_multi=input_multi.cuda()
     #for i in range(input_multi.shape[1]):
     input=input.cuda()#[:,i,:,:,:,:]
     B,N,C,H,W=input.shape
     input_copy=input.clone()
     out = orig = input.contiguous().view(-1,C,H,W)
     alpha = F.softmax(getattr(self, "alpha"), dim=-1)
     beta = F.softmax(getattr(self, "beta"), dim=-1)

     if self.prun_mode is not None:
            ratio = self.sample_prun_ratio(mode=self.prun_mode)
     else:
            ratio = self.sample_prun_ratio(mode=self._prun_modes)

     out = orig = self.conv_first(out)
     for i, cell in enumerate(self.cells_pre):
            out = cell(out, alpha[i], beta[i], ratio[i])
     out=out+orig
     L1_fea = out
     if self.align_type=="PCD":
         L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
         L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))

         L1_fea = L1_fea.view(B, N, -1, H, W)
         L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
         ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            ]
         aligned_fea = []
         for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone()]
            frame_al=self.align(nbr_fea_l, ref_fea_l)
            frame_al = torch.nn.functional.interpolate(frame_al, size=[8,8], mode="bilinear")
            aligned_fea.append(frame_al)
         aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
     elif self.align_type=="DKC":
         aligned_fea = []
         aligned_fea_hr=[]
         L1_fea= L1_fea.view(B, N, -1, H, W)
         ref_fea_l = L1_fea[:, self.center, :, :, :].clone()
         for i in range(N):
             nbr_fea_l = L1_fea[:, i, :, :, :].clone()
             frame_al=self.align(nbr_fea_l, ref_fea_l)
             #aligned_fea_hr.append(frame_al)
             #frame_al=torch.nn.functional.interpolate(frame_al, size=[8,8], mode="bilinear")
             aligned_fea.append(frame_al)
         aligned_fea = torch.stack(aligned_fea, dim=1)
         #aligned_fea_hr = torch.stack(aligned_fea_hr,dim=1)
     out=orig=aligned_fea
     alpha_attn_levels = F.softmax(getattr(self, "alpha_att_levels"), dim=-1)
     alpha_attn_activations = F.softmax(getattr(self, "alpha_activations"), dim=-1)
     alpha_sink=F.softmax(getattr(self,'alpha_sink'),dim=-1)
     for i, cell in enumerate(self.cells_attn):
            out = cell(out, alpha_attn_levels[i],alpha_sink[i],alpha_attn_activations[i])
     out = out + orig
     if self.prun_mode is not None:
        ratio = self.sample_prun_ratio(mode=self.prun_mode)
     else:
        ratio = self.sample_prun_ratio(mode=self._prun_modes)
     output_attn=out
     output_attn=output_attn.view(B,-1,H,W)
     out = orig = self.conv_attn_final(output_attn)
     for i, cell in enumerate(self.cells_recon):
            out = cell(out, alpha[i], beta[i], ratio[i])
     out=out+orig
     # TODO: add cells
     x=self.lrelu(self.conv11(out))
     x = self.lrelu(self.conv112(x))
     #x = self.pixshuff(x)+input_copy
     x = self.pixshuff(x)+F.interpolate(input_copy[:,self.center,:,:,:].clone(),scale_factor=2,mode='bicubic',align_corners=False)
     x = self.lrelu(self.conv12(x))
     
     x = self.lrelu(self.conv122(x))
     x = self.pixshuff(x)+F.interpolate(input_copy[:,self.center,:,:,:].clone(),scale_factor=4,mode='bicubic',align_corners=False)

     x = self.lrelu(self.conv13(x))
     x = self.lrelu(self.conv132(x))
     #x = self.pixshuff(x)

     x = self.pixshuff(x)+F.interpolate(input_copy[:,self.center,:,:,:].clone(),scale_factor=8,mode='bicubic',align_corners=False)
     x=self.lrelu(self.conv14(x))
     x = self.conv142(x)
     x = self.pixshuff(x)
     img_upscaled = F.interpolate(input_copy[:, self.center, :, :, :].clone(),scale_factor=16,mode='bicubic', align_corners=False)
     out = img_upscaled + x

     if ENABLE_TANH:
            out = (self.tanh(out) + 1) / 2
     #output.append(out)
     output=out #torch.stack(output,1)
     return output

    def forward_flops(self, size, alpha=True, beta=True, ratio=True):
        '''if alpha:
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = torch.ones_like(getattr(self, 'alpha')) * 1. / len(PRIMITIVES)

        if beta:
            beta = F.softmax(getattr(self, "beta"), dim=-1)
        else:
            beta = torch.ones_like(getattr(self, 'beta')) * 1. / 2

        if ratio:
            if self.prun_mode is not None:
                ratio = self.sample_prun_ratio(mode=self.prun_mode)
            else:
                ratio = self.sample_prun_ratio(mode=self._prun_modes)
        else:
            ratio = self.sample_prun_ratio(mode='max')

        flops_total = []

        flops, size = self.conv_first.forward_flops(size)
        flops_total.append(flops)

        for i, cell in enumerate(self.cells_pre):
            flops, size = cell.forward_flops(size, alpha[i], beta[i], ratio[i])
            flops_total.append(flops)
        size1 = [1,size[0], self.H, self.W]
        #print(size1)
        if self.align_type=="PCD":
         #self.conv_first(input)
         flops, size = self.fea_L2_conv1.forward_flops(size)
         flops_total.append(flops)
         flops, size = self.fea_L2_conv2.forward_flops(size)
         flops_total.append(flops)
         size1 = [1,size[0],self.H,self.W]
         size2 = [1,size[0], self.H//2, self.W//2]
         flops=self.align.forward_flops(size1,size2,"PCD")
         flops_total.append(flops*self.num_frames)
        elif self.align_type=="DKC":
         flops = self.align.forward_flops(size1,"DKC")
         flops_total.append(flops*self.num_frames)
        size1 = [1,self.num_frames,size[0],self.H,self.W]
        #print(size)
        for i, cell in enumerate(self.cells_recon):
            flops, size = cell.forward_flops(size, alpha[i], beta[i], ratio[i])
            flops_total.append(flops)
        #print(size)
        size=[size[0],size[1],size[2]]
        #print(size)
        flops, size = self.conv11.forward_flops(size)
        flops_total.append(flops)
        size=[size[0]//4,size[1],size[2]]
        flops, size = self.conv12.forward_flops(size)
        flops_total.append(flops)
        size=[size[0]//4,size[1],size[2]]
        flops, size = self.conv13.forward_flops(size)
        flops_total.append(flops)
        '''
        return 100 #sum(flops_total)

    def gram(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    def _criterion_image(self, y_hat, x):
        base_loss = self.base_weight * self.loss_func(y_hat, x)

        y_c_features = self.vgg(x)
        y_hat_features = self.vgg(y_hat)

        content_loss = self.content_weight * self.loss_func(y_c_features, y_hat_features)

        diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        tv_loss = self.tv_weight * (diff_i + diff_j)

        total_loss = base_loss + content_loss + tv_loss

        return total_loss

    def _criterion_video(self,x,y):
        loss=self.loss_func(x[:,self.center_HR,:,:,:],x[:,self.center_HR-1,:,:,:])
        loss=loss+self.loss_func(x[:,self.center_HR,:,:,:],x[:,self.center_HR+1,:,:,:])
        if self.shift_loss:
         if self.use_mean_var:
             loss = loss +torch.mean((x[:,self.center_HR,:,:,:]-x[:,self.center_HR-1,:,:,:])-(y[:,self.center_HR,:,:,:]-y[:,self.center_HR-1,:,:,:]))
             loss = loss + torch.mean((x[:, self.center_HR, :, :, :] - x[:, self.center_HR+1, :, :, :]) - (y[:, self.center_HR, :, :, :] - y[:, self.center_HR+1, :, :, :]))
             loss = loss +torch.var((x[:,self.center_HR,:,:,:]-x[:,self.center_HR-1,:,:,:])-(y[:,self.center_HR,:,:,:]-y[:,self.center_HR-1,:,:,:]))
             loss = loss + torch.var((x[:, self.center_HR, :, :, :] - x[:, self.center_HR+1, :, :, :]) - (y[:, self.center_HR, :, :, :] - y[:, self.center_HR+1, :, :, :]))
         else:
             loss=loss+self.loss_func(x[:,self.center_HR,:,:,:]-x[:,self.center_HR-1,:,:,:],y[:,self.center_HR,:,:,:]-y[:,self.center_HR-1,:,:,:])
             loss=loss+self.loss_func(x[:,self.center_HR,:,:,:]-x[:,self.center_HR+1,:,:,:],y[:,self.center_HR,:,:,:]-y[:,self.center_HR+1,:,:,:])
        #print("Video loss",loss)
        return loss

    def _loss(self, input, target, pretrain=False):
        loss = 0

        if pretrain is not True:
            # "random width": sampled by gambel softmax
            self.prun_mode = None
            #print("Calling")
            logit = self(input)
            #print("Logit_shape",logit.shape)
            #for i in range(logit.shape[0]):
            loss = loss + self._criterion_image(logit, target)
            #print(loss)
            #if self.video_loss:
            #    loss = loss + self._criterion_video(logit, target)
        if len(self._width_mult_list) > 1:
            self.prun_mode = "max"
            #print("Calling")
            logit = self(input)
            #for i in range(logit.shape[1]):
            loss = loss + self._criterion_image(logit, target)
                #print(loss)
            #if self.video_loss:
            #    loss = loss + self._criterion_video(logit, target)
            self.prun_mode = "min"
            #print("Calling")
            logit = self(input)
            #for i in range(logit.shape[1]):
            loss = loss + self._criterion_image(logit, target)
            #if self.video_loss:
            #    loss = loss + self._criterion_video(logit, target)
            #    #print(loss)
            if pretrain == True:
                self.prun_mode = "random"
                #print("Calling")
                logit = self(input)
                #for i in range(logit.shape[1]):
                loss = loss + self._criterion_image(logit, target)
                #print(loss)
                #if self.video_loss:
                #    loss = loss + self._criterion_video(logit, target)
                self.prun_mode = "random"
                #print("Calling")
                logit = self(input)
                #for i in range(logit.shape[1]):
                loss = loss + self._criterion_image(logit, target)
                #print(loss)
                #if self.video_loss:
                #    loss = loss + self._criterion_video(logit, target)
        elif pretrain == True and len(self._width_mult_list) == 1:
            self.prun_mode = "max"
            #print("Calling")
            logit = self(input)
            #for i in range(logit.shape[1]):
            loss = loss + self._criterion_image(logit, target)
            #print(loss)
            #if self.video_loss:
            #    loss = loss + self._criterion_video(logit, target)
        #print(loss)
        return loss

    def _build_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        num_ops_attention = len(PRIMITIVES_attention)
        setattr(self, 'alpha_sink', nn.Parameter(
            Variable(1e-3 * torch.ones(self.num_cell, self.op_per_cell_attn, 2*num_ops_attention), requires_grad=True)))
        setattr(self, 'alpha_activations', nn.Parameter(
            Variable(1e-3 * torch.ones(self.num_cell_attn, self.op_per_cell_attn, 2*num_ops_attention,3),
                     requires_grad=True)))
        setattr(self, 'alpha', nn.Parameter(
            Variable(1e-3 * torch.ones(self.num_cell, self.op_per_cell, num_ops), requires_grad=True)))
        setattr(self, 'alpha_att_levels', nn.Parameter(
            Variable(1e-3 * torch.ones(self.num_cell_attn,self.op_per_cell_attn,num_ops_attention, num_ops_attention), requires_grad=True)))
        setattr(self, 'beta', nn.Parameter(
            Variable(1e-3 * torch.ones(self.num_cell, self.op_per_cell, 2), requires_grad=True)))

        if self._prun_modes == 'arch_ratio':
            # prunning ratio
            num_widths = len(self._width_mult_list)
        else:
            num_widths = 1

        setattr(self, 'ratio', nn.Parameter(
            Variable(1e-3 * torch.ones(self.num_cell, self.op_per_cell - 1, num_widths), requires_grad=True)))

        return {"alpha_sink": self.alpha_sink,"alpha_att_levels":self.alpha_att_levels,"alpha_activations": self.alpha_activations,"alpha": self.alpha, "beta": self.beta, "ratio": self.ratio}

    def _reset_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        num_ops_attention =len(PRIMITIVES_attention)
        if self._prun_modes == 'arch_ratio':
            # prunning ratio
            num_widths = len(self._width_mult_list)
        else:
            num_widths = 1
        getattr(self,'alpha_activations').data = Variable(1e-3 * torch.ones(self.num_cell_attn,self.op_per_cell_attn,2*num_ops_attention,3),
                                               requires_grad=True)
        getattr(self, "alpha_sink").data = Variable(1e-3 * torch.ones(self.num_cell_attn, self.op_per_cell_attn, 2*num_ops_attention),
                                               requires_grad=True)
        getattr(self, "alpha_att_levels").data = Variable(1e-3 * torch.ones(self.num_cell_attn,self.op_per_cell_attn,num_ops_attention, num_ops_attention),
                                                   requires_grad=True)
        getattr(self, "alpha").data = Variable(1e-3 * torch.ones(self.num_cell, self.op_per_cell, num_ops),
                                               requires_grad=True)
        getattr(self, "beta").data = Variable(1e-3 * torch.ones(self.num_cell, self.op_per_cell, 2),
                                              requires_grad=True)
        getattr(self, "ratio").data = Variable(
            1e-3 * torch.ones(self.num_cell, self.op_per_cell - 1, num_widths), requires_grad=True)
