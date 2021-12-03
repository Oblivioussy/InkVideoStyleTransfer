from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect
import re
import scipy.ndimage
import numpy as np
import os, math
import cv2, skimage
import collections
import torchvision.transforms as transforms
import random


# read next frame from a capture
def getframe(cap):
    suc = False
    while not suc:
        suc, frame = cap.read()
    return frame


def transformbyopt(opt, imgs, seed=None, additional_scale=1):
    expend = False
    if len(imgs.shape) == 3:
        imgs = imgs[np.newaxis, :]
        expend = True
    resize = list(imgs.shape)
    if opt.resize_or_crop == 'resize_and_crop':
        resize[1] = resize[2] = opt.loadSize
    elif opt.resize_or_crop == 'scale_width':
        if resize[1] < resize[2]:
            resize[2] = round(resize[2] * opt.fineSize / resize[1])
            resize[1] = opt.fineSize
        else:
            resize[1] = round(resize[1] * opt.fineSize / resize[2])
            resize[2] = opt.fineSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        if resize[1] < resize[2]:
            resize[2] = round(resize[2] * opt.loadSize / resize[1])
            resize[1] = opt.loadSize
        else:
            resize[1] = round(resize[1] * opt.loadSize / resize[2])
            resize[2] = opt.loadSize
    if additional_scale != 1:
        resize[1] *= additional_scale
        resize[2] *= additional_scale
    resize[1] = math.ceil(resize[1]/64) * 64
    resize[2] = math.ceil(resize[2]/64) * 64
    tmp = list()
    for img in imgs:
        tmpi = skimage.transform.resize(img, (resize[1], resize[2]))
        tmp.append(tmpi)
    
    imgs = torch.from_numpy(np.stack(tmp)).permute((0, 3, 1, 2)).float()
    # imgs = torch.nn.functional.interpolate(
    #     imgs, size=resize[1:3], mode="nearest")
    imgs = imgs  / 127.5 - 1

    if opt.resize_or_crop.find("crop") > -1:
        bias1 = random.randint(0, imgs.shape[2] - opt.fineSize)
        bias2 = random.randint(0, imgs.shape[3] - opt.fineSize)
        if seed is None:
            seed = (bias1, bias2)
        imgs = imgs[:, :, bias1:bias1+opt.fineSize, bias2:bias2+opt.fineSize]
    if expend:
        imgs = imgs[0]
    return imgs, seed


def transformopticalflow(opt, imgs, seed):
    if len(imgs.shape) == 3:
        resize = [1., 1., 1.]
        if opt.resize_or_crop == 'resize_and_crop':
            resize[0] = float(opt.loadSize) / imgs.shape[0]
            resize[1] = float(opt.loadSize) / imgs.shape[1]
        elif opt.resize_or_crop == 'scale_width':
            resize[0] = resize[1] = float(opt.fineSize) / \
                max(imgs.shape[0], imgs.shape[1])
        elif opt.resize_or_crop == 'scale_width_and_crop':
            resize[0] = resize[1] = float(opt.loadSize) / \
                max(imgs.shape[0], imgs.shape[1])
        imgs = scipy.ndimage.zoom(imgs, resize)
        if opt.resize_or_crop == 'crop' or opt.resize_or_crop == 'scale_width_and_crop' or opt.resize_or_crop == 'resize_and_crop':
            bias1, bias2 = seed
            imgs = imgs[bias1:bias1+opt.fineSize, bias2:bias2+opt.fineSize]
        fix = np.array([[[resize[1], resize[0]]]])
        return torch.Tensor(imgs * fix)
    elif len(imgs.shape) == 4:
        resize = [1., 1., 1., 1.]
        if opt.resize_or_crop == 'resize_and_crop':
            resize[1] = float(opt.loadSize) / imgs.shape[1]
            resize[2] = float(opt.loadSize) / imgs.shape[2]
        elif opt.resize_or_crop == 'scale_width':
            resize[1] = resize[2] = float(opt.fineSize) / \
                max(imgs.shape[1], imgs.shape[2])
        elif opt.resize_or_crop == 'scale_width_and_crop':
            resize[1] = resize[2] = float(opt.loadSize) / \
                max(imgs.shape[1], imgs.shape[2])
            if resize[1] < opt.loadSize or resize[2] < opt.loadSize:
                resize[1] = resize[2] = float(opt.loadSize) / \
                    min(imgs.shape[1], imgs.shape[2])
        imgs = scipy.ndimage.zoom(imgs, resize)
        if opt.resize_or_crop == 'crop' or opt.resize_or_crop == 'scale_width_and_crop' or opt.resize_or_crop == 'resize_and_crop':
            bias1, bias2 = seed
            imgs = imgs[:, bias1:bias1+opt.fineSize, bias2:bias2+opt.fineSize]
        fix = np.array([[[[resize[2], resize[1]]]]])
        return torch.Tensor(imgs * fix)


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array


def tensor2im(image_tensor, imtype=np.uint8):
    # # print(image_tensor.shape)
    # # image_numpy = image_tensor[0].cpu().float().numpy()
    # # if image_numpy.shape[0] == 1:
    # #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    # # # print(image_numpy.shape)
    # # mean = np.zeros(image_numpy.shape)
    # # mean[0, :, :] = (0.485/0.229)
    # # mean[1, :, :] = (0.456/0.224)
    # # mean[2, :, :] = (0.406/0.225)
    # # std = mean
    # # std[0, :, :] = 0.229
    # # std[1, :, :] = 0.224
    # # std[2, :, :] = 0.225
    # # mean = mean.transpose((1, 2, 0))
    # # std = std.transpose((1, 2, 0))
    # # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + mean) * std * 255.0
    # flag = True
    # image_tensor =  image_tensor[0]
    # if image_tensor.shape[0]==1:
    #     image_tensor = torch.cat([image_tensor,image_tensor,image_tensor],dim = 0)
    #     flag = False
    #     dn_img = image_tensor
    # # print('image_tensor')
    # # print(image_tensor.shape)
    # else:
    #     denorm = transforms.Normalize(mean=[-1 * (0.485 / 0.229), -1 * (0.456 / 0.224), -1 * (0.406 / 0.225)],
    #                                   std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    #     dn_img = denorm(image_tensor)
    # # print('dn_img')
    # # print(dn_img.shape)
    # image_numpy = dn_img.cpu().float().numpy()
    # image_numpy[image_numpy<0]=0.0
    # image_numpy[image_numpy>1]=1.0
    # if flag==False:
    #     image_numpy = 1.0-image_numpy
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * 255.0)
    # # image_numpy = image_numpy.astype(np.uint8)
    # # image_pil = Image.fromarray(image_numpy)
    # # image_pil.save('./'+str(1)+'.jpg')
    # return image_numpy.astype(imtype)
    image_numpy = image_tensor.data.cpu().float().numpy()
    if len(image_numpy.shape) == 4:
        image_numpy = image_numpy[0]
    if len(image_numpy.shape) == 2:
        image_numpy = image_numpy[np.newaxis, :]
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def warp(x, flo):
    # print("warp check:", x.shape, flo.shape)
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W, 1)
    yy = yy.view(1, H, W, 1)
    grid = torch.cat((xx, yy), 3).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid = 2*vgrid / \
        torch.Tensor([W-1, H-1]).to(x.device).view(1, 1, 1, 2) - 1

    output = torch.nn.functional.grid_sample(x, vgrid, align_corners=True)
    return output
