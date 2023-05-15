import os
from os.path import basename
import math
import argparse
import random
import logging
import cv2

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from utils.timer import Timer, TickTock
from utils.util import get_resume_paths
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()
opt = option.parse(args.opt, is_train=True)
opt['dist'] = False
opt = option.dict_to_nonedict(opt)
model = create_model(opt,0)
s=torch.nn.Sigmoid()
LR=s(torch.randn([2,3,40,40]).double().type(torch.DoubleTensor).cuda())
HR=s(torch.randn([2,3,160,160]).double().type(torch.DoubleTensor).cuda())
data={}
data['LQ']=LR
data['GT']=HR
model.feed_data(data)
z, nll, y_logits = model.netG(gt=HR, lr=LR, reverse=False)
inp_rev,ldet=model.netG(lr=LR, z=z, eps_std=1, reverse=True)
#for p in model.netG.parameters():
#    print(p.name)
torch.sum(nll).backward()
for name, param in model.netG.named_parameters():
    if param.requires_grad:
        print(name)
print(inp_rev.shape)
print(HR.shape)
print(nll)
print(ldet)
print(inp_rev[0,1,0,0])
print(HR[0,1,0,0])
