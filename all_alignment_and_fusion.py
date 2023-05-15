import torch
from torch import nn as nn
from torch.nn import functional as F
from thop import profile
from basicsr.models.archs.arch_util import (DCNv2Pack, ResidualBlockNoBN,
                                            make_layer)
from thop import clever_format
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
from basicsr.models.archs.arch_util import DCNv2Pack as DCN
#try:
#    from models.archs.dcn.deform_conv import ModulatedDeformConvPack as DCN
#except ImportError:
#    raise ImportError('Failed to import DCNv2 module.')

try:
    from models.archs.deformable_kernels.modules import GlobalDeformKernel2d as GDK
    from models.archs.deformable_kernels.modules import DeformKernelConv2d as DKC
except ImportError:
    raise ImportError('Failed to import Deformable kernels')
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

class DK_spatial_attention_v2(nn.Module):
    '''Deformable kernel spatial attention module
    for last 3 x2 levels'''

    def __init__(self, nf=64):
        super(DK_spatial_attention_v2, self).__init__()

        self.DKC = nn.Sequential(nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1, bias=True),
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
                                 nn.Conv2d(nf, 4 * nf, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.PixelShuffle(2),
                                 nn.Conv2d(nf, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):

        return x * self.DKC(x)

class Align_fea(nn.Module):
    def __init__(self, nf=64,  groups=8):
        super(Align_fea, self).__init__()
        self.offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
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

class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

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
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3,
                                                     1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            self.dcn_pack[level] = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                padding=1,
                deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                  1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
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
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
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

#PCD pipeline feature_extractor->level extractor->PCD_Align levels
num_feat=64
conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1).cuda()
conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1).cuda()
conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1).cuda()
conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1).cuda()
pcd_align = PCDAlignment(
    num_feat=64, deformable_groups=8).cuda()
#Input random extracted features define
inp=torch.randn([32*5,64,32,32]).cuda()
b,t,c,h,w=32,5,64,32,32
feat_l1=inp
feat_l2 = conv_l2_1(feat_l1)
feat_l2 = conv_l2_2(feat_l2)
# L3
feat_l3 = conv_l3_1(feat_l2)
feat_l3 = conv_l3_2(feat_l3)
feat_l1 = feat_l1.view(b, t, -1, h, w)
feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)
#PCD
center_frame_idx=2
ref_feat_l = [  # reference feature list
    feat_l1[:, center_frame_idx, :, :, :].clone(),
    feat_l2[:, center_frame_idx, :, :, :].clone(),
    feat_l3[:, center_frame_idx, :, :, :].clone()
]
aligned_feat = []
for i in range(t):
    nbr_feat_l = [  # neighboring feature list
        feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
        feat_l3[:, i, :, :, :].clone()
    ]
    aligned_feat.append(pcd_align(nbr_feat_l, ref_feat_l))
aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)
macs, params = profile(pcd_align, inputs=(nbr_feat_l,ref_feat_l ))
macs, params = clever_format([macs, params], "%.3f")
print("PCD macs",macs)
print("PCD params",params)
align = Align_fea(nf=64, groups=8).cuda()
macs, params = profile(align, inputs=(feat_l1[:, i, :, :, :].contiguous(),feat_l1[:, center_frame_idx, :, :, :].contiguous() ))
macs, params = clever_format([macs, params], "%.3f")
print("DKC macs",macs)
print("DKC params",params)
input=torch.randn([32,5,64,32,32]).cuda()
cov=nn.Conv2d(64*5,64,3,1,1)
DKSA=DK_spatial_attention_v2(64).cuda()
out_conv=cov(input.view(32,-1,32,32))
macs1, params1 = profile(cov, inputs=(input ))
macs, params = profile(DKSA, inputs=(out_conv ))
macs, params = clever_format([macs+macs1, params+params1], "%.3f")
print("DKSA macs",macs)
print("DKSA params",params)
#input=torch.randn([32,5,64,32,32]).cuda()
EPAB=EPAB(64,5).cuda()
macs, params = profile(EPAB, inputs=(input))
macs, params = clever_format([macs, params], "%.3f")
print("EPAB macs",macs)
print("EPAB params",params)
