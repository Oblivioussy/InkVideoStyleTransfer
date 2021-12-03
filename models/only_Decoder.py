import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
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
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
sys.path.insert(
    1, "/home/linchpin/Documents/ink_stylize/ChipGAN_release/models/detectron2_repo/projects/PointRend")
import point_rend

def no_sigmoid_cross_entropy(sig_logits, label, mask=None):
    # print(sig_logits)
    if mask is not None:
        count_neg = torch.sum((1.-label)*mask)
        count_pos = torch.sum(label*mask)

        beta = count_neg / (count_pos+count_neg)
        pos_weight = beta / (1-beta)

        cost = pos_weight * label * \
            (-1) * torch.log(sig_logits) + \
            (1-label) * (-1) * torch.log(1-sig_logits)
        cost = torch.sum(cost * (1-beta) * mask) / \
            torch.sum(mask) / cost.shape[0] / cost.shape[1]
    else:
        count_neg = torch.sum((1.-label))
        count_pos = torch.sum(label)

        beta = count_neg / (count_pos+count_neg)
        pos_weight = beta / (1-beta)

        cost = pos_weight * label * \
            (-1) * torch.log(sig_logits) + \
            (1-label) * (-1) * torch.log(1-sig_logits)
        cost = torch.mean(cost * (1-beta))
    return cost
# ~~~~~~


def L2dis(v):
    return torch.sum(v*v, dim=1, keepdim=True)


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig +
                        interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        out_filter = np.array(kernel, dtype=np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis=2)
        return out_filter

    def gauss(self, mask):
        res = list()
        for i in range(mask.shape[1]):
            res.append(self.gauss_conv(mask[:, i:i+1]))
        return torch.cat(res, dim=1)

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.display_param = dict()

        self.long_term = [0, 1, 10, 20, 40]
        self.alpha1 = opt.alpha1
        self.alpha2 = opt.alpha2
        self.alpha = opt.alpha
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A_encoder = networks.define_G_encoder(opt.input_nc, opt.output_nc,
                                                        opt.ngf, opt, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, opt.saliency, opt.multisa)
        self.netG_A_decoder = networks.define_G_decoder(opt.input_nc, opt.output_nc,
                                                        opt.ngf, opt, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, opt.multisa)

        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        chs = self.netG_A_encoder.channel_size()
        self.netM = networks.define_convs(chs, 1, opt.M_layers, opt.M_size, gpu_ids=self.gpu_ids)
        self.netM2 = networks.define_convs(chs, 1, opt.M_layers, opt.M_size, gpu_ids=self.gpu_ids)
        # self.netInpaint = networks.define_convs(self.netG_A_encoder.channel_size(
        # ), self.netG_A_encoder.channel_size(), 3, opt.M_size, gpu_ids=self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            self.netD_ink = networks.define_D(opt.output_nc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            g_kernel = self.gauss_kernel(21, 3, 1).transpose((3, 2, 1, 0))
            self.gauss_conv = nn.Conv2d(
                1, 1, kernel_size=21, stride=1, padding=1, bias=False)
            self.gauss_conv.weight.data.copy_(torch.from_numpy(g_kernel))
            self.gauss_conv.weight.requires_grad = False
            self.gauss_conv.cuda()

            kw = 11
            g_kernel = self.gauss_kernel(kw, 3, 1).transpose((3, 2, 1, 0))
            self.gauss_conv_kw = nn.Conv2d(
                1, 1, kernel_size=kw, stride=1, padding=0, bias=False)
            self.gauss_conv_kw.weight.data.copy_(torch.from_numpy(g_kernel))
            self.gauss_conv_kw.weight.requires_grad = False
            self.gauss_conv_kw.cuda()
            self.gauss_conv_kw_pad = nn.ReflectionPad2d((kw-1)//2)

            L = np.array([1, 1]).reshape(2, 1)
            H = np.array([-1, 1]).reshape(2, 1)
            haar_kernel = np.stack(
                (L@(H.T), H@(L.T), H@(H.T))).reshape(3, 1, 2, 2) / 2
            self.haar_kernel = nn.Conv2d(
                1, 3, 2, stride=2, padding=0, bias=False)
            self.haar_kernel.weight.data.copy_(torch.from_numpy(haar_kernel))
            self.haar_kernel.weight.requires_grad = False
            self.haar_kernel.cuda()

            # ~~~~~~
            self.hed_model = Hed()
            self.hed_model.cuda()
            save_path = './pretrained_models/35.pth'
            self.hed_model.load_state_dict(torch.load(save_path))
            for param in self.hed_model.parameters():
                param.requires_grad = False
            # ~~~~~~

        # ~~~~~~
        if opt.saliency:
            cfg = get_cfg()
            point_rend.add_pointrend_config(cfg)
            cfg.merge_from_file(
                "/home/linchpin/Documents/ink_stylize/ChipGAN_release/models/detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.WEIGHTS = "pretrained_models/model_final_3c3198.pkl"
            self.NetIS = build_model(cfg)
            checkpointer = DetectionCheckpointer(self.NetIS)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.NetIS.eval()
            for param in self.NetIS.parameters():
                param.requires_grad = False

        self.pwc_model = PWCnet.PWCNet().cuda().eval()
        model_path = './pretrained_models/network-default.pytorch'
        self.pwc_model.load_state_dict(torch.load(model_path))
        for param in self.pwc_model.parameters():
            param.requires_grad = False

        # ~~~~~~

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A_encoder, 'G_A_encoder', which_epoch)
            self.load_network(self.netG_A_decoder, 'G_A_decoder', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            self.load_network(self.netM, "M", which_epoch)
            self.load_network(self.netM2, "M2", which_epoch)
            # self.load_network(self.netInpaint, "Inpaint", which_epoch)
        if opt.continue_train:
            self.load_network(self.netD_A, 'D_A', which_epoch)
            self.load_network(self.netD_B, 'D_B', which_epoch)
            self.load_network(self.netD_ink, 'D_ink', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.ink_fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(
                use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.TV_LOSS = networks.TVLoss()
            # initialize optimizers
            # , self.netInpaint.parameters()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A_decoder.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_M = torch.optim.Adam(itertools.chain(self.netM.parameters(
            ), self.netM2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(
                self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(
                self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_ink = torch.optim.Adam(
                self.netD_ink.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.continue_train:
                self.load_optim(self.optimizer_M, "M", which_epoch)
                # self.load_optim(self.optimizer_G, "G", which_epoch)
                self.load_optim(self.optimizer_D_A, "D_A", which_epoch)
                self.load_optim(self.optimizer_D_B, "D_B", which_epoch)
                self.load_optim(self.optimizer_D_ink, "D_ink", which_epoch)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_M)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            self.optimizers.append(self.optimizer_D_ink)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A_encoder)
        networks.print_network(self.netG_A_decoder)
        # networks.print_network(self.netInpaint)
        networks.print_network(self.netG_B)
        networks.print_network(self.netM)
        networks.print_network(self.netM2)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        frames = input['frames']
        input_B = input['B']

        if len(self.gpu_ids) > 0:
            frames = frames.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)

        self.frames = frames[0]
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.index = input['index']

    def get_sa(self, img):
        # input should be -1 ~ 1
        assert len(img.shape) == 3
        height, width = img.shape[1:]
        img = torch.stack((img[2], img[1], img[0]))
        inputs = {"image": (img+1)*127.5, "height": height, "width": width}
        y = self.NetIS([inputs])[0]["instances"]._fields
        mask = y["pred_masks"]
        classes = y["pred_classes"]
        cnt = 0
        nsa = torch.zeros_like(self.frames[:1, :1])
        for t in range(classes.shape[0]):
            if classes[t] == 17:
                cnt += 1
                assert cnt < 10

        tmp = 0
        for t in range(classes.shape[0]):
            if classes[t] == 17:
                tmp += 1
                nsa = nsa + mask[t:t+1].float().unsqueeze(0) * tmp / cnt
        box = y['pred_boxes'].tensor.long()
        if (box.shape[0] == 0):
            box = (-1, -1, -1, -1)
        else:
            box = (torch.min(box[:, 0]), torch.min(box[:, 1]),
                   torch.max(box[:, 2]), torch.max(box[:, 3]))
        return nsa, box

    def scale_to(self, a, b, c, d, base=4, maxstep=4):
        a -= a % base
        c += (base-(c % base)) % base
        b -= b % base
        d += (base-(d % base)) % base
        for _ in range(maxstep):
            if a > 0:
                a -= base
            if b > 0:
                b -= base
            if c < self.frames.shape[3]:
                c += base
            if d < self.frames.shape[2]:
                d += base
        return a, b, c, d

    def forward(self):
        with torch.no_grad():
            kernel_size = 5
            pad_size = kernel_size//2
            p1d = (pad_size, pad_size, pad_size, pad_size)
            p_real_B = nn.functional.pad(self.input_B, p1d, "constant", 1)
            erode_real_B = -1 * \
                (nn.functional.max_pool2d(-1*p_real_B, kernel_size, 1))

            res1 = self.gauss_conv(erode_real_B[:, 0, :, :].unsqueeze(1))
            res2 = self.gauss_conv(erode_real_B[:, 1, :, :].unsqueeze(1))
            res3 = self.gauss_conv(erode_real_B[:, 2, :, :].unsqueeze(1))

            self.ink_real_B = torch.cat((res1, res2, res3), dim=1)
            if self.frames.shape[0] > 1:
                optF = PWCnet.estimate(self.pwc_model, self.frames[0:1].repeat(
                    self.frames.shape[0]-1, 1, 1, 1), self.frames[1:])
                toptF = optF.permute(0, 3, 1, 2)
                tmp = nn.functional.interpolate(
                    toptF, size=(self.frames.shape[2] // 4, self.frames.shape[3] // 4), mode="bilinear")
                tmp[:, 0, :, :] = tmp[:, 0, :, :] * \
                    tmp.shape[3] / toptF.shape[3]
                tmp[:, 1, :, :] = tmp[:, 1, :, :] * \
                    tmp.shape[2] / toptF.shape[2]
                tFoptF = tmp
                FoptF = tmp.permute(0, 2, 3, 1)

                noptF = PWCnet.estimate(self.pwc_model, self.frames[1:], self.frames[0:1].repeat(
                    self.frames.shape[0]-1, 1, 1, 1))
                tnoptF = noptF.permute(0, 3, 1, 2)
                tmp = nn.functional.interpolate(
                    tnoptF, size=(self.frames.shape[2] // 4, self.frames.shape[3] // 4), mode="bilinear")
                tmp[:, 0, :, :] = tmp[:, 0, :, :] * \
                    tmp.shape[3] / tnoptF.shape[3]
                tmp[:, 1, :, :] = tmp[:, 1, :, :] * \
                    tmp.shape[2] / tnoptF.shape[2]
                tFnoptF = tmp
                FnoptF = tmp.permute(0, 2, 3, 1)

                C = util.warp(tnoptF, optF)
                tmp = C + toptF
                C = (L2dis(tmp) < self.alpha1 *
                     (L2dis(C) + L2dis(toptF)) + self.alpha2).float()

                self.optF = optF
                self.toptF = toptF
                self.tFoptF = tFoptF
                self.FoptF = FoptF
                self.noptF = noptF
                self.tFnoptF = tFnoptF
                self.FnoptF = FnoptF
                self.C = C
                del tmp

            if self.opt.saliency:
                sa = list()
                box = list()
                for img in self.frames:
                    a, b = self.get_sa(img)
                    if torch.sum(a) < 0.001:
                        return False
                    sa.append(a)
                    box.append(torch.stack(b))
                    # exit(0)
                self.SA = torch.cat(sa, dim=0)
                self.sa = sa[0][0].data.cpu()
                self.box = box
                # print(self.sa.shape)
                if self.frames.shape[0] > 1:
                    self.warped_sa = util.warp(self.SA[1:], self.optF)
        return True

    def netG_A(self, x, sa=None):
        if self.opt.saliency:
            assert sa is not None
        return self.netG_A_decoder(self.netG_A_encoder(x, sa), sa)

    def test(self):
        # self.netG_A_decoder.eval()
        # self.netG_A_encoder.eval()
        # self.netM.eval()
        # inked = set()
        # uninked = set(range(self.frames.shape[0]))
        # res = torch.zeros_like(self.frames)

        # def L2dis(v):
        #     return torch.sum(v*v, dim=1)

        # def calc_dis(A, B):
        #     optF = PWCnet.estimate(self.pwc_model, A, B)
        #     return torch.mean(torch.sqrt(L2dis(optF)))

        # dis = dict()
        # for i in range(self.frames.shape[0]):
        #     for j in range(i+1, self.frames.shape[0]):
        #         dis[(i, j)] = calc_dis(self.frames[i:i+1], self.frames[j:j+1])
        #         dis[(j, i)] = dis[(i, j)]

        # def evalue(x):
        #     val1 = 0
        #     for i in inked:
        #         val1 += dis[(x, i)]
        #     val2 = 0
        #     for i in uninked:
        #         if i != x:
        #             val2 += dis[(x, i)]
        #     return val1 / (len(inked)+1) - self.alpha * val2 / (len(uninked)+1)

        # while len(uninked) > 0:
        #     idx = 0
        #     mn = 1e30
        #     for i in uninked:
        #         tmp = evalue(i)
        #         if tmp < mn:
        #             mn = tmp
        #             idx = i
        #     l = list()
        #     for i in inked:
        #         l.append((dis[(idx, i)], i))
        #     l.sort()
        #     if len(l) > 5:
        #         l = l[:5]
        #     l = [self.frames[idx]]+list(map(lambda x: self.frames[x[1]], l))
        #     now = torch.stack(l)
        #     if self.opt.saliency:
        #         sa, _, _, _, _, _, _, _ = self.bas_model(now)
        #         sa = sa[:, 0:1, :, :]
        #         mn = torch.min(torch.min(sa, dim=2, keepdim=True).values,
        #                        dim=3, keepdim=True).values
        #         mx = torch.max(torch.max(sa, dim=2, keepdim=True).values,
        #                        dim=3, keepdim=True).values
        #         sa = (sa-mn) / (mx - mn)
        #     F = self.netG_A_encoder(
        #         torch.cat((now, sa), dim=1) if self.opt.saliency else now)

        #     optF = PWCnet.estimate(self.pwc_model, now[0:1].repeat(
        #         now.shape[0]-1, 1, 1, 1), now[1:])
        #     toptF = optF.permute(0, 3, 1, 2)
        #     tmp = nn.functional.interpolate(toptF, size=F.shape[2:], mode="bilinear")
        #     tmp[:, 0, :, :] = tmp[:, 0, :, :] * tmp.shape[3] / optF.shape[3]
        #     tmp[:, 1, :, :] = tmp[:, 1, :, :] * tmp.shape[2] / optF.shape[2]
        #     FoptF = tmp.permute(0, 2, 3, 1)

        #     F = torch.cat((F[:1], util.warp(F[1:], FoptF)), dim=0)
        #     score = torch.softmax(self.netM(F-F[0:1]), dim=0)
        #     oF = torch.sum(F * score, dim=0, keepdim=True)
        #     res[idx:idx+1] = self.netG_A_decoder(oF)

        #     inked.add(idx)
        #     uninked.remove(idx)
        return res

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # print(loss_D_fake.item(), loss_D_real.item())
        # backward
        loss_D.backward()
        return loss_D.item()

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(
            self.netD_A, self.input_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(
            self.netD_B, self.input_A, fake_A)

    def backward_D_ink(self):
        ink_fake_B = self.ink_fake_B_pool.query(self.ink_fake_B)
        self.loss_D_ink = self.backward_D_basic(
            self.netD_ink, self.ink_real_B, ink_fake_B)

    def clac_GA_common_loss(self, fake_B, lambda_sup, SA=None, edge=True, lambda_newloss=1):
        # GAN Loss
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True) * 1.5
        self.loss_G_A = loss_G_A.item()
        # print(loss_G_A.item())

        # ink loss
        kernel_size = 5
        pad_size = kernel_size//2
        p1d = (pad_size, pad_size, pad_size, pad_size)
        p_fake_B = nn.functional.pad(fake_B, p1d, "constant", 1)
        erode_fake_B = -1 * \
            (nn.functional.max_pool2d(-1*p_fake_B, kernel_size, 1))

        res1 = self.gauss_conv(erode_fake_B[:, 0, :, :].unsqueeze(1))
        res2 = self.gauss_conv(erode_fake_B[:, 1, :, :].unsqueeze(1))
        res3 = self.gauss_conv(erode_fake_B[:, 2, :, :].unsqueeze(1))

        ink_fake_B = torch.cat((res1, res2, res3), dim=1)
        pred_fake_ink = self.netD_ink(ink_fake_B)
        loss_G_ink = self.criterionGAN(
            pred_fake_ink, True) * self.opt.lambda_ink
        self.loss_G_ink = loss_G_ink.item()
        self.ink_fake_B = ink_fake_B.detach()
        del res1, res2, res3, erode_fake_B, p_fake_B
        if SA is not None and torch.sum(SA) > 0.1:
            # border loss
            if abs(self.opt.lambda_border * lambda_newloss) > 1e-5:
                haar_SA = torch.abs(self.haar_kernel(SA))
                cmask = torch.sum(haar_SA, dim=1).unsqueeze(1)
                assert cmask.shape[1] == 1
                cmask = (cmask > 0.0001).float()
                cmask = self.gauss_conv_kw(self.gauss_conv_kw_pad(cmask))

                gray_fakeB = fake_B[:, :1, :, :] * 0.3 + fake_B[:,
                                                                1:2, :, :] * 0.59 + fake_B[:, 2:, :, :] * 0.11
                haar_B = torch.abs(self.haar_kernel(gray_fakeB))
                loss_border = 2 - torch.sum(haar_B*cmask)/torch.sum(cmask)
                self.tv = cmask.data.cpu()


                # haar_SA = self.haar_kernel(SA)
                # cmask = torch.sum(torch.abs(haar_SA), dim=1).unsqueeze(1)
                # assert cmask.shape[1] == 1
                # cmask = (cmask > 0.0001).float()
                # cmask = self.gauss_conv_kw(self.gauss_conv_kw_pad(cmask))

                # gray_fakeB = fake_B[:, :1, :, :] * 0.3 + fake_B[:,
                #                                                 1:2, :, :] * 0.59 + fake_B[:, 2:, :, :] * 0.11
                # haar_B = self.haar_kernel(gray_fakeB)
                # loss_border = 2 - torch.sum(-haar_B*haar_SA * cmask)/torch.sum(cmask)

                # self.tv = haar_SA.data.cpu()

                loss_border *= self.opt.lambda_border * lambda_newloss
                self.loss_border = loss_border.item()
                del haar_B, cmask, gray_fakeB, haar_SA
            else:
                self.loss_border = loss_border = 0
            emask = self.gauss_conv_kw(
                self.gauss_conv_kw_pad((SA > 0.0001).float()))
            if edge:
                # edge loss
                edge_real_A = torch.sigmoid(
                    self.hed_model(self.input_A).detach())
                edge_fake_B = torch.sigmoid(self.hed_model(fake_B))

                loss_edge_1 = no_sigmoid_cross_entropy(
                    edge_fake_B, edge_real_A * (emask * 0.8+0.2), mask=emask*0.8+0.2) * lambda_sup
                self.edge_real_A = (edge_real_A).data.cpu()
                self.edge_fake_B = (edge_fake_B).data.cpu()

                self.loss_edge_1 = loss_edge_1.item()
            else:
                self.loss_edge_1 = loss_edge_1 = 0
            # back_loss
            bmask = 1-emask
            if abs(self.opt.lambda_back * lambda_newloss) > 1e-5:
                tmp = torch.pow((1-fake_B), 2)
                loss_back = torch.sum(tmp * bmask) / \
                    torch.sum(bmask) * (0.3 + 0.8 / (torch.sum(tmp * emask) / \
                    torch.sum(emask)))
                loss_back *= self.opt.lambda_back * lambda_newloss
                self.loss_back = loss_back.item()

                # tmp = torch.pow((1-fake_B), 2)
                # tmp2 = torch.pow((fake_B+1), 2)
                # loss_back = torch.sum(tmp * bmask) / \
                #     torch.sum(bmask) / tmp.shape[0]/tmp.shape[1]
                # loss_back += 0.1 * \
                #     torch.sum(tmp2 * emask) / torch.sum(emask) / \
                #     tmp.shape[0]/tmp.shape[1]
                # loss_back *= self.opt.lambda_back * lambda_newloss
                # self.loss_back = loss_back.item()
            else:
                self.loss_back = loss_back = 0
        else:
            self.loss_border = loss_border = 0
            self.loss_edge_1 = loss_edge_1 = 0
            self.loss_back = loss_back = 0
        if abs(self.opt.lambda_TV) > 0:
            loss_TV = self.TV_LOSS(fake_B)
            loss_TV *= self.opt.lambda_TV
            self.loss_TV = loss_TV.item()
        else:
            self.loss_TV = loss_TV = 0
        del emask, bmask, edge_fake_B, edge_real_A
        return loss_back + loss_edge_1 + loss_border + loss_G_A + loss_G_ink + loss_TV

    def getF(self, indexA=0, indexB=None, rev=False):
        with torch.no_grad():
            if indexB is None:
                indexB = self.frames.shape[0]
            F_horses = list()
            SA_horse = list()
            frame = self.frames[indexA:indexB] if not rev else 0-self.frames[indexA:indexB]
            frame = frame # + (self.SA[indexA:indexB]<0.00001).float() * 0.2
            if self.opt.saliency:
                if self.opt.inpaint:
                    F = self.netG_A_encoder(
                        frame, torch.zeros_like(self.SA[indexA:indexB]))
                    F_all = F.clone()
                    pos = list()
                    for i in range(indexA, indexB):
                        a, b, c, d = [x.item() for x in self.box[i]]
                        a, b, c, d = self.scale_to(a, b, c, d)
                        sa_horse = self.SA[i:i+1, :, b:d, a:c]
                        frame_horse = frame[i-indexA:i+1-indexA, :, b:d, a:c]
                        h, w = d-b, c-a
                        sc = min(self.frames.shape[2]/h, self.frames.shape[3]/w)
                        nh = h*sc
                        nw = w*sc
                        nh = math.ceil(nh/4)*4
                        nw = math.ceil(nw/4)*4
                        frame_horse = nn.functional.interpolate(
                            frame_horse, size=(nh, nw), mode="bilinear")
                        sa_horse_scaled = nn.functional.interpolate(
                            sa_horse, size=(nh, nw), mode="nearest")
                        F_horse = self.netG_A_encoder(frame_horse, sa_horse_scaled)
                        aa = a//4
                        bb = b//4
                        F_horse_scaled = nn.functional.interpolate(
                            F_horse, size=(h//4, w//4), mode="bilinear")
                        cc = aa+F_horse_scaled.shape[3]
                        dd = bb+F_horse_scaled.shape[2]
                        assert cc == c//4 and dd == d//4

                        sa_horse = nn.functional.interpolate(
                            (sa_horse > 0.0001).float(), size=(dd-bb, cc-aa), mode="nearest")
                        F_all[i-indexA:i-indexA+1, :,
                            bb:dd, aa:cc] = (F_horse_scaled * sa_horse) + F_all[i-indexA:i-indexA+1, :, bb:dd, aa:cc] * (1-sa_horse)
                        F_horses.append(F_horse)
                        SA_horse.append(sa_horse_scaled)
                        pos.append((a, b, c, d))
                        del sa_horse, sa_horse_scaled, F_horse, F_horse_scaled, frame_horse
                    del F
                    # return self.netInpaint(F_all), F_horses, SA_horse, pos
                    return F_all, F_horses, SA_horse, pos
                else:
                    return self.netG_A_encoder(self.frames[indexA:indexB], self.SA[indexA:indexB])
            else:
                return self.netG_A_encoder(self.frames[indexA:indexB])

    def backward_G(self, lambda_sup, lambda_newloss):
        self.input_A = self.frames[0:1]
        SA = self.SA[:1]

        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # ink
        if self.opt.inpaint:
            F, F_horse, SA_horse, pos = self.getF()
            # F2, _, _, _ = self.getF(0, 1, rev=True)
        else:
            F = self.getF()
            # F2 = self.getF(0, 1, rev=True)

        if self.opt.saliency:
            inked = self.netG_A_decoder(F, self.SA)
            fake_B = inked[:1]
            # fake_rev = self.netG_A_decoder(F2, self.SA[:1])

            # Crop Loss
            if self.opt.inpaint and abs(self.opt.lambda_crop * lambda_newloss) > 0:
                ink_horse = self.netG_A_decoder(F_horse[0], SA_horse[0])
                a, b, c, d = pos[0]

                ink_horse = nn.functional.interpolate(
                    ink_horse, size=(d-b, c-a), mode="bilinear")
                self.boxed = ink_horse

                loss_crop = torch.mean(torch.pow(
                    fake_B[:, :, b:d, a:c] - ink_horse, 2)) * self.opt.lambda_crop * lambda_newloss
                self.loss_crop = loss_crop.item()
            else:
                loss_crop = self.loss_crop = 0
        else:
            inked = self.netG_A_decoder(F)
            fake_B = inked[:1]
            fake_rev = self.netG_A_decoder(F2)
            loss_crop = self.loss_crop = 0
        self.fake_B = fake_B.detach()

        # loss rev
        # loss_rev = torch.mean(torch.pow(fake_B-fake_rev, 2)) * self.opt.lambda_rev
        # self.loss_rev = loss_rev.item()

        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B(self.input_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(
            pred_fake, True)
        self.loss_G_B = loss_G_B.item()
        self.fake_A = fake_A

        # Forward cycle loss
        rec_A = self.netG_B(fake_B)
        loss_cycle_A = self.criterionCycle(
            rec_A, self.input_A) * lambda_A
        self.loss_cycle_A = loss_cycle_A.item()
        self.rec_A = rec_A.data.cpu()
        del rec_A

        # Backward cycle loss
        if self.opt.saliency:
            B_sa = list()
            for img in fake_A:
                B_sa.append(self.get_sa(img)[0])
            B_sa = torch.cat(B_sa, dim=0)
            self.B_sa = B_sa[0].data.cpu()
        else:
            self.loss_class = 0
        if self.opt.saliency:
            rec_B = self.netG_A(fake_A, B_sa)
        else:
            rec_B = self.netG_A(fake_A)
        del B_sa
        loss_cycle_B = self.criterionCycle(
            rec_B, self.input_B) * lambda_B
        self.loss_cycle_B = loss_cycle_B.item()
        self.rec_B = rec_B.data.cpu()
        del rec_B, fake_A

        # color loss
        # loss_color = torch.mean(torch.pow(self.netG_A(-self.input_A, self.SA[:1]) - fake_B, 2))

        # Warp loss
        if abs(self.opt.lambda_warp * lambda_newloss) > 0 and self.frames.shape[0] > 1 and torch.mean(self.C) > 0.1:
            I = util.warp(inked[1:2], self.optF[:1])
            if self.opt.saliency:
                II = self.netG_A_decoder(
                    util.warp(F[1:2], self.FoptF[:1]), self.warped_sa[:1])
            else:
                II = self.netG_A_decoder(util.warp(F[1:2], self.FoptF[:1]))
            dI = I - II
            warp_loss = torch.sum(dI * dI * self.C[:1]) / torch.sum(self.C[:1])
            warp_loss *= self.opt.lambda_warp * lambda_newloss
            self.warp_loss = warp_loss.item()
            self.fake_warp_error = dI[-1].data.cpu()
            self.warp_error = (util.warp(self.frames[1:2], self.optF[:1])-self.frames[:1]).data.cpu()
            del I, II, dI
        else:
            warp_loss = 0
            if not hasattr(self, "warp_loss"):
                self.warp_loss = 0
            if not hasattr(self, "warp_error"):
                self.warp_error = torch.zeros((3, 1, 1))
        del F

        loss = loss_cycle_A + loss_G_B + loss_cycle_B + warp_loss +\
            self.clac_GA_common_loss(
                fake_B, lambda_sup, self.SA[:1], lambda_newloss=lambda_newloss)
        loss.backward()
        if torch.isnan(loss).item():
            raise ValueError
        # if self.opt.saliency:
        #     del sa_horse_scaled, sa_horse, F_horse, pred_fake

    def backward_M(self, lambda_sup, lambda_newloss):
        lambda_temporal = self.opt.lambda_temporal
        self.input_A = self.frames[0:1]
        temporal_loss = 0

        F, _, _, _ = self.getF()
        FoptF = self.FoptF
        optF = self.optF
        C = self.C

        # temporal loss
        oF = util.warp(F[1:], FoptF)
        tmp = list()
        for f in F[1:]:
            tmp.append(self.netM((f-F[0]).unsqueeze(0)))
            # tmp.append(self.netM(torch.cat((f,F[0]), dim=0).unsqueeze(0)))
        tmp = torch.cat(tmp, dim=0)
        score = torch.softmax(tmp, dim=0)
        oF = torch.sum(oF * score, dim=0, keepdim=True)
        self.display_param["score1"] = score.data.cpu()
        # score = (torch.tanh(self.netM2(torch.abs(oF - F[:1]))) + 1) / 2
        score = (torch.tanh(self.netM2(oF-F[:1])) + 1) / 2
        self.score = torch.mean(score[0].data.cpu(), dim=0)
        oF = score * oF + (1-score) * F[:1]

        assert oF.shape == F[:1].shape

        if self.opt.saliency:
            frames = self.netG_A_decoder(F, self.SA)
        else:
            frames = self.netG_A_decoder(F)
        I = torch.cat((frames[:1], util.warp(frames[1:], optF)), dim=0)

        if self.opt.saliency:
            O = self.netG_A_decoder(oF, self.SA[:1])
        else:
            O = self.netG_A_decoder(oF)
        self.M_fake_B = O.cpu()
        self.fake_B = I[:1].cpu()
        nC = torch.zeros(O.shape[2:]).to(
            device=I.device).unsqueeze(0).unsqueeze(0)

        for i in range(1, F.shape[0]):
            mask = (C[i-1:i] > nC + 0.1).float()
            nC = nC + mask
            dI = I[i:i+1] - O
            temporal_loss += torch.sum((C[i-1:i]+0.01) * torch.pow(dI,2)) / torch.sum(C[i-1:i]+0.01)
            print("{}: {}".format(i, temporal_loss.item()))
        dI = I[0:1] - O
        temporal_loss += torch.sum((1-nC+0.01) * torch.pow(dI,2)) / torch.sum(1-nC+0.01)
        print("temp loss:", temporal_loss.item())
        temporal_loss *= lambda_temporal / self.frames.shape[0]

        # smooth_loss
        # loss_smooth = self.TV_LOSS(score)
        # loss_smooth *= self.opt.lambda_smooth
        # self.loss_smooth = loss_smooth.item()
        self.loss_smooth = loss_smooth = 0

        self.mask = nC[0].data.cpu()
        print("check score2:", torch.mean(
            score[-1]).item(), torch.max(score[-1]).item())
        self.display_param["score2"] = score[-1].data.cpu()

        self.temporal_loss = temporal_loss.item()
        self.occlusion = nC.data[0].cpu()

        # mask loss
        if abs(self.opt.lambda_occ) > 0:
            loss_occ = torch.mean(torch.pow(
                score-nn.functional.interpolate(nC, score.shape[2:], mode="bilinear"), 2))
            loss_occ *= self.opt.lambda_occ
            self.loss_occ = loss_occ.item()
        else:
            self.loss_occ = loss_occ = 0

        # score temporal loss
        if self.frames.shape[0] > 2:
            oF = util.warp(F[2:3], FoptF[1:2])
            score = (torch.tanh(self.netM2(oF-F[:1])) + 1) / 2

            noptF = PWCnet.estimate(self.pwc_model, self.frames[1:2], self.frames[2:3])
            tnoptF = noptF.permute(0, 3, 1, 2)
            tmp = nn.functional.interpolate(
                tnoptF, size=(self.frames.shape[2] // 4, self.frames.shape[3] // 4), mode="bilinear")
            tmp[:, 0, :, :] = tmp[:, 0, :, :] * \
                tmp.shape[3] / tnoptF.shape[3]
            tmp[:, 1, :, :] = tmp[:, 1, :, :] * \
                tmp.shape[2] / tnoptF.shape[2]
            tmp = tmp.permute(0, 2, 3, 1)

            oF2 = util.warp(F[2:3], tmp)
            score2 = (torch.tanh(self.netM2(oF2-F[1:2])) + 1) / 2
            score2 = util.warp(score2, FoptF[:1])
            tC = nn.functional.interpolate(self.C[:1], score.shape[2:], mode="bilinear")
            ds = torch.abs(score2-score)
            mask = (ds < 0.5).float() * tC

            loss_score_temp = torch.sum(ds*mask) / torch.sum(mask)
            loss_score_temp *= self.opt.lambda_score_temp
            self.loss_score_temp = loss_score_temp.item()
        else:
            self.loss_score_temp = loss_score_temp = 0

        total_loss = temporal_loss + loss_occ + loss_smooth + loss_score_temp\
            + self.clac_GA_common_loss(
                O, lambda_sup, self.SA[:1], edge=True, lambda_newloss=lambda_newloss)
        total_loss.backward()
        if torch.isnan(total_loss).item():
            raise ValueError
        del nC, dI, I, FoptF, optF, F, oF, score

    def optimize_parameters(self, lambda_sup, lambda_newloss=1):
        torch.cuda.empty_cache()
        # forward
        if self.forward():
            if self.opt.trainM <= epoch and self.frames.shape[0] > 1:
                self.optimizer_M.zero_grad()
                if self.opt.trainGAN:
                    self.optimizer_G.zero_grad()
                self.backward_M(lambda_sup, lambda_newloss)
                self.optimizer_M.step()
                if self.opt.trainGAN:
                    self.optimizer_G.step()
            if self.opt.trainGAN:
                # G_A and G_B
                if self.frames.shape[0] > 2:
                    self.frames = self.frames[:2]
                    self.SA = self.SA[:2]
                    self.optF = self.optF[:1]
                    self.FoptF = self.FoptF[:1]
                self.optimizer_G.zero_grad()
                self.backward_G(lambda_sup, lambda_newloss)
                self.optimizer_G.step()
                if random.randint(0,2) == 0:
                    # D_A
                    self.optimizer_D_A.zero_grad()
                    self.backward_D_A()
                    self.optimizer_D_A.step()
                    # D_B
                    self.optimizer_D_B.zero_grad()
                    self.backward_D_B()
                    self.optimizer_D_B.step()
                    # D_ink
                    self.optimizer_D_ink.zero_grad()
                    self.backward_D_ink()
                    self.optimizer_D_ink.step()

    def loadattr(self, dic, name, attr, func=None):
        if hasattr(self, attr):
            attr = getattr(self, attr)
            if func is not None:
                attr = func(attr)
            dic[name] = attr

    def get_current_errors(self):
        ret_errors = OrderedDict()
        self.loadattr(ret_errors, "D_A", "loss_D_A")
        self.loadattr(ret_errors, "G_A", "loss_G_A")
        self.loadattr(ret_errors, "Cyc_A", "loss_cycle_A")
        self.loadattr(ret_errors, "D_B", "loss_D_B")
        self.loadattr(ret_errors, "G_B", "loss_G_B")
        self.loadattr(ret_errors, "Cyc_B", "loss_cycle_B")
        self.loadattr(ret_errors, "D_ink", "loss_D_ink")
        self.loadattr(ret_errors, "edge1", "loss_edge_1")
        self.loadattr(ret_errors, "G_ink", "loss_G_ink")
        self.loadattr(ret_errors, "Warp", "warp_loss")
        self.loadattr(ret_errors, "TV", "loss_TV")
        self.loadattr(ret_errors, "border", "loss_border")
        self.loadattr(ret_errors, "back", "loss_back")
        self.loadattr(ret_errors, "score_smooth", "loss_smooth")
        self.loadattr(ret_errors, "Occlusion", "loss_occ")
        self.loadattr(ret_errors, "Temporal", "temporal_loss")
        self.loadattr(ret_errors, "Crop", "loss_crop")
        self.loadattr(ret_errors, "m_edge", "m_edge_loss")
        self.loadattr(ret_errors, "loss_score_temp", "loss_score_temp")
        return ret_errors

    def get_current_visuals(self):
        ret = OrderedDict()
        self.loadattr(ret, "rec_B", "rec_B", util.tensor2im)
        self.loadattr(ret, "real_A", "input_A", util.tensor2im)
        self.loadattr(ret, "fake_A", "fake_A", util.tensor2im)
        self.loadattr(ret, "fake_B", "fake_B", util.tensor2im)
        self.loadattr(ret, "M_fake_B", "M_fake_B", util.tensor2im)
        self.loadattr(ret, "rec_A", "rec_A", util.tensor2im)
        self.loadattr(ret, "warp_err", "warp_error", util.tensor2im)
        self.loadattr(ret, "tv", "tv", util.tensor2im)
        self.loadattr(ret, "score", "score", util.tensor2im)
        self.loadattr(ret, "false_point", "mask", util.tensor2im)
        self.loadattr(ret, "ink_real_B", "ink_real_B", util.tensor2im)
        self.loadattr(ret, "real_B", "input_B", util.tensor2im)
        self.loadattr(ret, "edge_fake_B", "edge_fake_B", util.tensor2im)
        self.loadattr(ret, "edge_real_A", "edge_real_A", util.tensor2im)
        self.loadattr(ret, "ink_fake_B", "ink_fake_B", util.tensor2im)
        self.loadattr(ret, "SA", "sa", util.tensor2im)
        self.loadattr(ret, "B_SA", "B_sa", util.tensor2im)
        self.loadattr(ret, "boxed", "boxed", util.tensor2im)
        self.loadattr(ret, "boxed2", "boxed2", util.tensor2im)
        return ret

    def save(self, label):
        self.save_network(self.netG_A_encoder,
                          'G_A_encoder', label, self.gpu_ids)
        self.save_network(self.netG_A_decoder,
                          'G_A_decoder', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netM, 'M', label, self.gpu_ids)
        self.save_network(self.netM2, 'M2', label, self.gpu_ids)
        # self.save_network(self.netInpaint, 'Inpaint', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netD_ink, 'D_ink', label, self.gpu_ids)
        self.save_optim(self.optimizer_M, "M", label, self.gpu_ids)
        self.save_optim(self.optimizer_G, "G", label, self.gpu_ids)
        self.save_optim(self.optimizer_D_A, "D_A", label, self.gpu_ids)
        self.save_optim(self.optimizer_D_B, "D_B", label, self.gpu_ids)
        self.save_optim(self.optimizer_D_ink, "D_ink", label, self.gpu_ids)
