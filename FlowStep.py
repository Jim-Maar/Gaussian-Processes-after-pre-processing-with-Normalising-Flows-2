# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import torch
from torch import nn as nn

import models.modules
import models.modules.Permutations
from models.modules import flow, thops, FlowAffineCouplingsAblation
from utils.util import opt_get
from models.modules.spatial_attn import  _Spatial_first_order_attn #, Transformer_attn
from models.modules.transformer import Transformer_attn
from models.modules.elementwise_attention import Elementwise_channel_exp
def getConditional(rrdbResults, position):
    img_ft = rrdbResults if isinstance(rrdbResults, torch.Tensor) else rrdbResults[position]
    return img_ft


class FlowStep(nn.Module):
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "squeeze_invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_alternating_2_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlign": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1SubblocksShuf": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder4": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0, flow_permutation="invconv", flow_coupling="additive",
                 LU_decomposed=False, opt=None, image_injector=None, idx=None, acOpt=None, normOpt=None, in_shape=None,
                 position=None,inpshape=None):
        # check configures
        assert flow_permutation in FlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.image_injector = image_injector

        self.norm_type = normOpt['type'] if normOpt else 'ActNorm2d'
        self.position = normOpt['position'] if normOpt else None

        self.in_shape = in_shape
        self.position = position
        self.acOpt = acOpt
        #if flow_coupling == "CondAffineSeparatedAndCond":
        #self.attn =Transformer_attn(in_channels)
        # 1. actnorm
        self.actnorm = models.modules.FlowActNorms.ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = models.modules.Permutations.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)

        # 3. coupling
        if flow_coupling == "CondAffineSeparatedAndCond":
            self.affine = models.modules.FlowAffineCouplingsAblation.CondAffineSeparatedAndCond(in_channels=in_channels,
                                                                                                opt=opt)
        elif flow_coupling == "noCoupling":
            pass
        else:
            raise RuntimeError("coupling not Found:", flow_coupling)
        # 4. Attention
        #print(inpshape)
        #self.attn= Transformer_attn(in_channels)
        #self.attn_mask_elem1= Elementwise_channel_exp(in_channels)
        #self.attn_mask_elem1.init_mask(inpshape)
        #self.attn_mask_elem2= Elementwise_channel_exp(in_channels)
        #self.attn_mask_elem2.init_mask(inpshape)
        #self.attn_mask_elem3= Elementwise_channel_exp(in_channels)
        #self.attn_mask_elem3.init_mask(inpshape)       
        #self.attn_mask_elem4= Elementwise_channel_exp(in_channels)
        #self.attn_mask_elem4.init_mask(inpshape)
        #self.attn_mask_true_elem = Elementwise_channel_exp(in_channels)
        self.attn_mask_false_spa=_Spatial_first_order_attn(in_channels)
        self.attn_mask_true_spa= _Spatial_first_order_attn(in_channels)
        

    def forward(self, input, logdet=None, reverse=False, rrdbResults=None):
        if not reverse:
            return self.normal_flow(input, logdet, rrdbResults)
        else:
            return self.reverse_flow(input, logdet, rrdbResults)

    def normal_flow(self, z, logdet, rrdbResults=None):
        if self.flow_coupling == "bentIdentityPreAct":
            z, logdet = self.bentIdentPar(z, logdet, reverse=False)
        #print("Preact",logdet)
        # 1. actnorm
        #z, logdet = self.attn(input=z,logdet=logdet,reverse=False)
        if self.norm_type == "ConditionalActNormImageInjector":
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.actnorm(z, img_ft=img_ft, logdet=logdet, reverse=False)
        elif self.norm_type == "noNorm":
            pass
        else:
            z, logdet = self.actnorm(z, logdet=logdet, reverse=False)
        #z, logdet = self.attn(input=z,logdet=logdet,reverse=False)
        #print("Actnorm",logdet)
        # 2. permute
        #z, logdet = self.attn(input=z,logdet=logdet,reverse=False)
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)

        #z, logdet = self.attn_mask_false_spa(z,logdet=logdet,reverse=False,permute=False)
        #z, logdet = self.attn_mask_true_spa(z,logdet=logdet,reverse=False,permute=True)
        #print("Attn logdet",logdet)
        need_features = self.affine_need_features()
        #print("Permute",logdet)
        # 3. coupling
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affine(input=z, logdet=logdet, reverse=False, ft=img_ft)
        # 4. attention
        
        #if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
        #z, logdet = self.attn(input=z,logdet=logdet,reverse=False)
        #print("Coupling",logdet)
        #z, logdet = self.attn(input=z,logdet=logdet,reverse=False)
        #z, logdet = self.attn_mask_elem1(input=z,logdet=logdet,reverse=False)
        #z, logdet = self.attn_mask_elem2(z,logdet=logdet,reverse=False)
        z, logdet = self.attn_mask_true_spa(z,logdet=logdet,reverse=False,permute=False)
        z, logdet = self.attn_mask_false_spa(z,logdet=logdet,reverse=False,permute=True)
        #print("Attn logdet",logdet)
        return z, logdet

    def reverse_flow(self, z, logdet, rrdbResults=None):

        need_features = self.affine_need_features()
        #if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
        #z, logdet = self.attn_mask_true_elem(input=z,logdet=logdet,reverse=True,permute=False)
        #z, logdet = self.attn(input=z,logdet=logdet,reverse=True)
        z, logdet = self.attn_mask_false_spa(input=z,logdet=logdet,reverse=True,permute=True)
        z, logdet = self.attn_mask_true_spa(input=z,logdet=logdet,reverse=True,permute=False)
        #z, logdet = self.attn_mask_elem2(input=z,logdet=logdet,reverse=True)
        #z, logdet = self.attn_mask_elem1(input=z,logdet=logdet,reverse=True)
        
        #if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
        # 1.coupling
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affine(input=z, logdet=logdet, reverse=True, ft=img_ft)

        #z,logdet = self.attn_mask_true_spa(z,logdet=logdet,reverse=True,permute=True)
        #z,logdet = self.attn_mask_false_elem(z,logdet=logdet,reverse=True,permute=False)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)

        # 3. actnorm
        #z, logdet = self.attn(input=z,logdet=logdet,reverse=True)
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        # 4. attention
        #z,logdet = self.attn_mask_false_elem(z,logdet=logdet,reverse=True)
        #z,logdet =self.attn_mask_false_spa(z,logdet=logdet,reverse=True,permute=False)
        #z, logdet = self.attn(input=z,logdet=logdet,reverse=True)
        return z, logdet

    def affine_need_features(self):
        need_features = False
        try:
            need_features = self.affine.need_features
        except:
            pass
        return need_features
