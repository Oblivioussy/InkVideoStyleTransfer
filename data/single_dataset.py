import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, is_image_file
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import util.util as util
import torch, math


class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, "testA")
        self.A_paths = make_dataset(self.dir_A)
        B = []
        for i in self.A_paths:
            if i.find(".npy") == -1:
                B.append(i)
        self.A_paths = sorted(B)
        self.transform = get_transform(opt)
        self.splited = list()
        for i in range(len(self.A_paths)):
            if is_image_file(self.A_paths[i]):
                self.splited.append((self.A_paths[i], 0, 1))
            else:
                cap = cv2.VideoCapture(self.A_paths[i])
                cnt = int(cap.get(7))
                cap.release()
                n = math.ceil(cnt / 60)
                n = min(round(cnt / n), 65)

                ################################
                n = cnt

                now = 0
                while now < cnt:
                    if now > 0:
                        exit(0)
                    tmp = now + n
                    if cnt - now < 70 or cnt - now <= n:
                        tmp = cnt
                    elif cnt - now < 2 * n:
                        tmp = (cnt-now)//2 + now
                    self.splited.append((self.A_paths[i], now, tmp))
                    now = tmp

    def __getitem__(self, index):
        index_A = index
        A_path, l, r = self.splited[index_A]
        print("load", A_path)
        if is_image_file(A_path):
            img = Image.open(A_path)
            frames = np.array(img)
            frames = frames.astype(np.float32)[np.newaxis,:]
            frames, _ = util.transformbyopt(self.opt, frames)
        else:
            cap=cv2.VideoCapture(A_path)
            frames = list()
            cnt = int(cap.get(7))
            seed = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, l)
            for _ in range(min(r-l,10000)):
                __, f =cap.read()
                if __:
                    f = f[:,:,-1::-1].astype(np.float32)
                    tmpa, tmpb = util.transformbyopt(self.opt, f, seed, additional_scale=0.7 if (A_path.find("_8")!=-1 or A_path.find("_16")!=-1 or A_path.find("_17")!=-1) else 1)
                    frames.append(tmpa)
                    seed = tmpb
                else:
                    print("read error {}/{}frame in {}".format(_ + l, cnt, A_path))
                del f
            cap.release()
            frames = torch.stack(frames).float()
        d = dict()

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            frames = frames[:, 0:1, ...] * 0.299 + frames[
                :, 1:2, ...] * 0.587 + frames[:, 2:3, ...] * 0.114
        print("load finish", A_path)
        print("load:", frames.shape)
        # print(first.shape, opticalflow.shape)
        d['frames'] = frames
        tmp = os.path.splitext(A_path)
        # d['A_paths'] = (tmp[0], '_{}_{}'.format(l,r), tmp[1])
        d['A_paths'] = (tmp[0], '', tmp[1])
        d['index'] = index
        return d

    def __len__(self):
        return len(self.splited)

    def name(self):
        return 'SingleImageDataset'
