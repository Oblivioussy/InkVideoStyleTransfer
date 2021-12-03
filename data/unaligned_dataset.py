import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, is_image_file
from PIL import Image
import PIL
import random
from PIL import ImageFile
import numpy as np
import cv2
import torch
import util.util as util
import pickle
import bisect
import math
ImageFile.LOAD_TRUNCATED_IMAGES = True


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.long_term = [0, 2, 4] #, 8]
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A_video')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        # self.A_paths = list(filter(lambda x : random.randint(0,2)==0 or x.find('.avi') != -1, self.A_paths))
        self.A_length = [0]
        self.trans = get_transform(opt)
        for i in self.A_paths:
            if is_image_file(i):
                self.A_length.append(self.A_length[-1]+1)
            else:
                cap = cv2.VideoCapture(i)
                self.A_length.append(
                    self.A_length[-1]+math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT)/10))
        self.A_size = self.A_length[-1]
        self.B_size = len(self.B_paths)

    def __getitem__(self, index):
        d = dict()
        index_A = bisect.bisect_right(
            self.A_length, index % self.A_length[-1]) - 1
        A_path = self.A_paths[index_A]
        if is_image_file(A_path):
            print("load image from {}".format(A_path))
            img = np.array(Image.open(A_path))
            img, _ = util.transformbyopt(self.opt, img.astype(np.float32))
            frames = torch.unsqueeze(img, 0)
        else:
            frame_num = (
                index % self.A_length[-1] - self.A_length[index_A]) * 10 + random.randint(0, 9)
            print("load {} frame from {}".format(frame_num, A_path))
            cap = cv2.VideoCapture(A_path)
            if frame_num > cap.get(7):
                frame_num = int(cap.get(7))
            frames = list()
            seed = None
            for i in self.long_term:
                if i <= frame_num:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - i)
                    _, f = cap.read()
                    f, seed = util.transformbyopt(self.opt, f[:, :, -1::-1].astype(np.float32), seed)
                    frames.append(f)
            cap.release()
            frames = torch.stack(frames)
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_img = np.array(Image.open(B_path)).astype(np.float32)
        # print("load B")
        B, _ = util.transformbyopt(self.opt, B_img)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            first = first[0:1, ...] * 0.299 + first[
                1:2, ...] * 0.587 + first[2:3, ...] * 0.114
            second = second[0:1, ...] * 0.299 + second[
                1:2, ...] * 0.587 + second[2:3, ...] * 0.114

        if output_nc == 1:  # RGB to gray
            B = B[:, 0:1, ...] * 0.299 + B[:, 1:2, ...] * \
                0.587 + B[:, 2:3, ...] * 0.114
        # print(first.shape, opticalflow.shape)

        d['frames'] = frames
        d['B'] = B
        d['A_paths'] = A_path
        d['B_paths'] = B_path
        d['index'] = index
        return d

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
