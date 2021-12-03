import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from util.util import tensor2im
import cv2
import numpy as np
import torch
from data.image_folder import is_image_file
from PIL import Image
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
# test
for i, data in enumerate(dataset):
    print(i)
    if i >= opt.how_many:
        break
    model.set_input(data)
    with torch.no_grad():
        out = model.test()
    if isinstance(out, bool):
        print(model.get_image_paths()[0], "Failed")
    else:
        out, boxed, sc = out
        all_path = model.get_image_paths()
        img_path = all_path[0] + all_path[2]
        fname = os.path.split(img_path)[-1]

        img = ((model.frames.data.cpu().permute(
            (0, 2, 3, 1)).numpy() + 1) * 127.5).astype(np.uint8)

        if is_image_file(img_path):
            outname = os.path.splitext(fname)[0]+"_ink.png"
            im = Image.fromarray(out[0])
            im.save(os.path.join(
                opt.results_dir, outname))

            if boxed is not None:
                outname = os.path.splitext(fname)[0]+"_boxed.png"
                im = Image.fromarray(boxed[0])
                im.save(os.path.join(
                    opt.results_dir, outname))

            im = Image.fromarray(img[0])
            im.save(os.path.join(
                opt.results_dir, fname))
        else:
            cap = cv2.VideoCapture(img_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            forcc = cv2.VideoWriter_fourcc(*"mp4v")
            cap.release()

            img_path = all_path[0] + all_path[1] + all_path[2]
            fname = os.path.split(img_path)[-1]
            outname = os.path.splitext(fname)[0]+"_ink.mp4"
            size = (out.shape[2], out.shape[1])
            outv = cv2.VideoWriter(os.path.join(
                opt.results_dir, outname), forcc, fps, size)
            for x in range(out.shape[0]):
                outv.write(cv2.cvtColor(out[x], cv2.COLOR_RGB2BGR))
            outv.release()

            if boxed is not None:
                outname = os.path.splitext(fname)[0]+"_boxed.mp4"
                size = (boxed.shape[2], boxed.shape[1])
                boxedv = cv2.VideoWriter(os.path.join(
                    opt.results_dir, outname), forcc, fps, size)
                for x in range(boxed.shape[0]):
                    boxedv.write(cv2.cvtColor(boxed[x], cv2.COLOR_RGB2BGR))
                boxedv.release()
            
            if sc is not None:
                outname = os.path.splitext(fname)[0]+"_score.mp4"
                size = (sc.shape[2], sc.shape[1])
                scv = cv2.VideoWriter(os.path.join(
                    opt.results_dir, outname), forcc, fps, size)
                for x in range(sc.shape[0]):
                    scv.write(cv2.cvtColor(sc[x], cv2.COLOR_RGB2BGR))
                scv.release()

            size = (img.shape[2], img.shape[1])
            imgv = cv2.VideoWriter(os.path.join(
                opt.results_dir, os.path.splitext(fname)[0]+".mp4"), forcc, fps, size)
            for x in range(img.shape[0]):
                imgv.write(cv2.cvtColor(img[x], cv2.COLOR_RGB2BGR))
            imgv.release()
        print('%04d: process video... %s' % (i, img_path))

    del model.frames

    if i >= opt.how_many - 1:
        break
