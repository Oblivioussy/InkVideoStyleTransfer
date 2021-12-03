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

class CycleGAN(BaseModel):
    def name(self):
        return 'CycleGAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.display_param = dict()

        self.netG_A_encoder = networks.define_G_encoder(opt.input_nc, opt.output_nc,
                                                        opt.ngf, opt, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, False, False)
        self.netG_A_decoder = networks.define_G_decoder(opt.input_nc, opt.output_nc,
                                                        opt.ngf, opt, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, False)

        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        # ~~~~~~

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A_encoder, 'G_A_encoder', which_epoch)
            self.load_network(self.netG_A_decoder, 'G_A_decoder', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
        if opt.continue_train:
            self.load_network(self.netD_A, 'D_A', which_epoch)
            self.load_network(self.netD_B, 'D_B', which_epoch)

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
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A_encoder.parameters(), self.netG_A_decoder.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(
                self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(
                self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.continue_train:
                self.load_optim(self.optimizer_G, "G", which_epoch)
                self.load_optim(self.optimizer_D_A, "D_A", which_epoch)
                self.load_optim(self.optimizer_D_B, "D_B", which_epoch)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A_encoder)
        networks.print_network(self.netG_A_decoder)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        frames = input['frames']
        if self.opt.isTrain:
            input_B = input['B']

        if len(self.gpu_ids) > 0:
            frames = frames.cuda(self.gpu_ids[0], async=True)
            if self.opt.isTrain:
                input_B = input_B.cuda(self.gpu_ids[0], async=True)

        self.frames = frames[0]
        if self.opt.isTrain:
            self.input_B = input_B
            self.image_paths = input['A_paths' if AtoB else 'B_paths']
        else:
            self.image_paths = (input['A_paths'][0][0],input['A_paths'][1][0],input['A_paths'][2][0])
        self.index = input['index']

    def forward(self):
        pass

    def netG_A(self, x, sa=None):
        return self.netG_A_decoder(self.netG_A_encoder(x, sa), sa)

    def test(self):
        ret = list()
        for i in range(self.frames.shape[0]):
            ret.append(util.tensor2im(self.netG_A_decoder(self.netG_A_encoder(self.frames[i:i+1]))))
        return np.stack(ret), None, None

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

    def clac_GA_common_loss(self, fake_B, lambda_sup, SA=None, edge=True, lambda_newloss=1):
        # GAN Loss
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)
        self.loss_G_A = loss_G_A.item()
        return loss_G_A

    def getF(self, indexA=0, indexB=None, rev=False):
        return self.netG_A_encoder(self.frames[indexA:indexB])

    def backward_G(self, lambda_sup, lambda_newloss):
        self.input_A = self.frames[0:1]

        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        F = self.getF()

        inked = self.netG_A_decoder(F)
        fake_B = inked[:1]
        self.fake_B = fake_B.detach()

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
        rec_B = self.netG_A(fake_A)
        loss_cycle_B = self.criterionCycle(
            rec_B, self.input_B) * lambda_B
        self.loss_cycle_B = loss_cycle_B.item()
        self.rec_B = rec_B.data.cpu()
        del rec_B, fake_A

        loss = loss_cycle_A + loss_G_B + loss_cycle_B +\
            self.clac_GA_common_loss(
                fake_B, lambda_sup)
        loss.backward()

    def optimize_parameters(self, lambda_sup, lambda_newloss=1, epoch=0):
        torch.cuda.empty_cache()
        # forward
        self.optimizer_G.zero_grad()
        self.backward_G(lambda_sup, lambda_newloss)
        self.optimizer_G.step()

        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

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
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_optim(self.optimizer_G, "G", label, self.gpu_ids)
        self.save_optim(self.optimizer_D_A, "D_A", label, self.gpu_ids)
        self.save_optim(self.optimizer_D_B, "D_B", label, self.gpu_ids)
