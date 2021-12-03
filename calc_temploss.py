import numpy as np
import torch
import os, cv2
import util.util as util
from util.image_pool import ImagePool
import sys
import math
import scipy.stats as st
import numpy as np
import torch.nn as nn
# ~~~~~~
from models.model import Hed
import models.PWCNet as PWCnet
import random
import matplotlib.pyplot as plt

def L2dis(v):
    return torch.sum(v*v, dim=1, keepdim=True)

pwc_model = PWCnet.PWCNet().cuda().eval()
model_path = './pretrained_models/network-default.pytorch'
pwc_model.load_state_dict(torch.load(model_path))
for param in pwc_model.parameters():
    param.requires_grad = False

d = 'results/CVPR19-Linear'
frame_cnt = 0
all_loss = 0

with torch.no_grad():
    for fn in os.listdir(d):
        if fn.find('ink') >= 0 and fn.find('merge') == -1 and (fn.find('mp4') >= 0 or fn.find('avi') >= 0):
            cap = cv2.VideoCapture(os.path.join(d, fn))
            frame_num = int(cap.get(7))
            frame_cnt += frame_num
            frames = list()
            for i in range(frame_num):
                _, f = cap.read()
                f = torch.from_numpy(f).permute(2,0,1).float()
                frames.append(f)
            cap.release()
            ink_frames = torch.stack(frames).cuda()
            
            cap = cv2.VideoCapture(os.path.join(d, fn))
            frames = list()
            for i in range(frame_num):
                _, f = cap.read()
                f = torch.from_numpy(f).permute(2,0,1).float()
                frames.append(f)
            cap.release()
            frames = torch.stack(frames).cuda()
            now = 0
            temploss = 0

            if frames.shape[0] > ink_frames.shape[0]:
                frames = frames[:ink_frames.shape[0]]
            if frames.shape[0] < ink_frames.shape[0]:
                ink_frames = ink_frames[:frames.shape[0]]
            if frames.shape[2] != ink_frames.shape[2]:
                ink_frames = nn.functional.interpolate(ink_frames, frames.shape[2:])
            
            # od = torch.randperm(frames.shape[0])
            # frames = frames[od]
            # ink_frames = ink_frames[od]

            while now < frames.shape[0] - 1:
                end = min(frames.shape[0] - 1, now + 30)
                optF = PWCnet.estimate(pwc_model, frames[now:end]/127.5-1, frames[now+1:end+1]/127.5-1)
                toptF = optF.permute(0, 3, 1, 2)

                noptF = PWCnet.estimate(pwc_model, frames[now+1:end+1]/127.5-1, frames[now:end]/127.5-1)
                tnoptF = noptF.permute(0, 3, 1, 2)

                C = util.warp(tnoptF, optF)
                tmp = C + toptF
                C = (L2dis(tmp) < 0.01 *
                        (L2dis(C) + L2dis(toptF)) + 0.5).float()
                
                wframe = util.warp(ink_frames[now+1:end+1], optF)

                temploss += (torch.sum(torch.abs((wframe-ink_frames[now:end]))*C) / torch.sum(C)).item()
                now = end

                del optF, toptF, noptF, tnoptF, wframe, C, tmp
                # print(now,'/',frames.shape[0])
            print(fn, temploss/frame_num)
            all_loss += temploss
            del frames, ink_frames

print(all_loss/frame_cnt, frame_cnt)