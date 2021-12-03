import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from scipy.misc import imresize
from torch.utils import tensorboard
import torch


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.name = opt.name
        self.opt = opt
        self.saved = False
        self.step = 0
        self.writer = tensorboard.SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "tensorboard"))

        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        util.mkdirs([self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        for label, image_numpy in visuals.items():
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            util.save_image(image_numpy, img_path)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, global_steps):
        message = '(epoch: %d, iters: %d, total_step: %d time: %.3f) ' % (epoch, i, global_steps, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
            self.writer.add_scalar(k, v, global_step=global_steps)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write(message+'\n')
    def print_display_param(self, para, global_steps):
        for k, v in para.items():
            if isinstance(v, float) or isinstance(v, int):
                self.writer.add_scalar(k, v, global_step=global_steps)
            elif isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                self.writer.add_histogram(k, v, global_step=global_steps)
        
    # save image to the disk
    def save_images(self, webpage, visuals, image_path, aspect_ratio=1.0):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        for label, im in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            if aspect_ratio < 1.0:
                im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            util.save_image(im, save_path)
