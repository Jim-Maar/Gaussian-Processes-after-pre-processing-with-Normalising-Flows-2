import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import arch_util as arch_util
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


class MDVR_CA_SA(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(MDVR_CA_SA, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA

        ResidualCA_Block_noBN_f = functools.partial(arch_util.ResidualCA_Block_noBN, nf=nf)
        Residual_Block_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(Residual_Block_noBN_f, front_RBs)
        # self.DKC_spatial_attention0 = arch_util.DK_spatial_attention(nf)

        # Alignment for each scale
        self.align = Align_fea(nf=nf, groups=groups)

        # fuse aligned features
        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        #### reconstruction
        # x2
        self.recon_trunk1 = arch_util.make_layer(ResidualCA_Block_noBN_f, 5)
        self.DKC_spatial_attention1 = arch_util.DK_spatial_attention(nf)
        self.HRconv_l1 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last_l1 = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        # x2
        self.conv_first2 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction2 = arch_util.make_layer(Residual_Block_noBN_f, 1)
        self.recon_trunk2 = arch_util.make_layer(ResidualCA_Block_noBN_f, 3)
        self.DKC_spatial_attention2 = arch_util.DK_spatial_attention_v2(nf)
        self.HRconv_l2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last_l2 = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        # x2
        '''self.conv_first3 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction3 = arch_util.make_layer(Residual_Block_noBN_f, 1)
        self.recon_trunk3 = arch_util.make_layer(ResidualCA_Block_noBN_f, 15)
        self.DKC_spatial_attention3 = arch_util.DK_spatial_attention_v2(nf)
        self.HRconv_l3 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last_l3 = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        # x2
        self.conv_first4 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction4 = arch_util.make_layer(Residual_Block_noBN_f, 1)
        self.recon_trunk4 = arch_util.make_layer(ResidualCA_Block_noBN_f, 10)
        self.DKC_spatial_attention4 = arch_util.DK_spatial_attention_v2(nf)
        self.HRconv_l4 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last_l4 = nn.Conv2d(64, 3, 3, 1, 1, bias=True)'''
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, 64 * 4*4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4*4, 3, 1, 1, bias=True)
        '''self.upconv3 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.upconv4 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)'''
        self.pixel_shuffle = nn.PixelShuffle(4)
        # self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        # self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()
        print(x.shape)
        #### extract LR features
        fea_ori = self.lrelu(self.conv_first(x.contiguous().view(-1, C, H, W)))
        fea = self.feature_extraction(fea_ori)
        # fea = self.DKC_spatial_attention0(fea)
        # fea += fea_ori

        ############# align level 1  #################
        fea = fea.view(B, N, -1, H, W)
        # ref feature list
        ref_fea_l = fea[:, self.center, :, :, :].clone()
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = fea[:, i, :, :, :].clone()
            aligned_fea.append(self.align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        # print(aligned_fea.shape)
        aligned_fea = aligned_fea.view(B, -1, H, W)
        fea = self.fea_fusion(aligned_fea)
        ###Level1##
        # recons + up x2
        out1 = self.recon_trunk1(fea)
        out1 = self.DKC_spatial_attention1(out1)
        out1 += fea
        out1 = self.lrelu(self.pixel_shuffle(self.upconv1(out1)))
        out1 = self.lrelu(self.HRconv_l1(out1))
        out1 = self.conv_last_l1(out1)
        out1 += F.interpolate(x_center, scale_factor=4, mode='bicubic', align_corners=False)

        ################### level 2  ##############

        # recons + up x2
        out2 = self.conv_first2(out1)
        out2_res = self.feature_extraction2(out2)
        out2 = self.recon_trunk2(out2_res)
        out2 = self.DKC_spatial_attention2(out2)
        out2 += out2_res
        out2 = self.lrelu(self.pixel_shuffle(self.upconv2(out2)))
        out2 = self.lrelu(self.HRconv_l2(out2))
        out2 = self.conv_last_l2(out2)
        out2 += F.interpolate(out1, scale_factor=4, mode='bicubic', align_corners=False)

        #############  level 3   ############
        '''

        out3 = self.conv_first3(out2)
        out3_res = self.feature_extraction3(out3)
        out3 = self.recon_trunk3(out3_res)
        out3 = self.DKC_spatial_attention3(out3)
        out3 += out3_res
        out3 = self.lrelu(self.pixel_shuffle(self.upconv3(out3)))
        out3 = self.lrelu(self.HRconv_l3(out3))
        out3 = self.conv_last_l3(out3)
        out3 += F.interpolate(out2, scale_factor=2, mode='bicubic', align_corners=False)

        #############  level 4   ############

        out4 = self.conv_first4(out3)
        out4_res = self.feature_extraction4(out4)
        out4 = self.recon_trunk4(out4_res)
        out4 = self.DKC_spatial_attention4(out4)
        out4 += out4_res
        out4 = self.lrelu(self.pixel_shuffle(self.upconv4(out4)))
        out4 = self.lrelu(self.HRconv_l4(out4))
        out4 = self.conv_last_l4(out4)
        out4 += F.interpolate(out3, scale_factor=2, mode='bicubic', align_corners=False)
        '''
        return out2
