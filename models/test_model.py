from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
import scipy.stats as st
from .base_model import BaseModel
from . import networks
import numpy as np
import torch, cv2, math
import torch.nn as nn
import models.PWCNet as PWCnet
from PIL import Image
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
import sys
sys.path.insert(1, "/home/linchpin/Documents/ink_stylize/ChipGAN_release/models/detectron2_repo/projects/PointRend")
import point_rend


def L2dis(v):
    return torch.sum(v*v, dim=1)

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

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
            res.append(self.gauss_conv_kw(mask[:,i:i+1]))
        return torch.cat(res, dim=1)

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.long_term = [0, 1, 10, 20, 40]
        self.alpha1 = opt.alpha1
        self.alpha2 = opt.alpha2
        self.alpha = opt.alpha
        # load/define networks
        self.netG_A_encoder = networks.define_G_encoder(opt.input_nc, opt.output_nc,
                                                        opt.ngf, opt, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, opt.saliency, opt.multisa)
        self.netG_A_decoder = networks.define_G_decoder(opt.input_nc, opt.output_nc,
                                                        opt.ngf, opt, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, opt.multisa)

        self.netM = networks.define_convs(self.netG_A_encoder.channel_size(
        ) *2, 1, opt.M_layers, opt.M_size, gpu_ids=self.gpu_ids)
        self.netM2 = networks.define_convs(self.netG_A_encoder.channel_size(
        ) *2, 1, opt.M_layers, opt.M_size, gpu_ids=self.gpu_ids)
        # ~~~~~~
        if opt.saliency:
            cfg = get_cfg()
            point_rend.add_pointrend_config(cfg)
            cfg.merge_from_file("/home/linchpin/Documents/ink_stylize/ChipGAN_release/models/detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml") 
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.WEIGHTS = "/home/linchpin/Documents/ink_stylize/ChipGAN_release/pretrained_models/model_final_3c3198.pkl"
            self.NetIS = build_model(cfg)
            checkpointer = DetectionCheckpointer(self.NetIS)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.NetIS.eval()
            if len(self.opt.gpu_ids) == 0:
                self.NetIS.cpu()
            for param in self.NetIS.parameters():
                param.requires_grad = False
            

        self.pwc_model = PWCnet.PWCNet().eval()
        if len(self.opt.gpu_ids) != 0:
            self.pwc_model.cuda()
        model_path = './pretrained_models/network-default.pytorch'
        self.pwc_model.load_state_dict(torch.load(model_path))
        for param in self.pwc_model.parameters():
            param.requires_grad = False

        # ~~~~~~
        kw = 3
        g_kernel = self.gauss_kernel(kw, 3, 1).transpose((3, 2, 1, 0))
        self.gauss_conv_kw = nn.Conv2d(
            1, 1, kernel_size=kw, stride=1, padding=(kw-1)//2, bias=False)
        self.gauss_conv_kw.weight.data.copy_(torch.from_numpy(g_kernel))
        self.gauss_conv_kw.weight.requires_grad = False
        if len(self.opt.gpu_ids) != 0:
            self.gauss_conv_kw.cuda()

        which_epoch = opt.which_epoch
        self.load_network(self.netG_A_encoder, 'G_A_encoder', which_epoch)
        self.load_network(self.netG_A_decoder, 'G_A_decoder', which_epoch)
        self.load_network(self.netM, "M", which_epoch)
        self.load_network(self.netM2, "M2", which_epoch)
        # self.netG_A_decoder.eval()
        # self.netG_A_encoder.eval()
        # self.netM.eval()
        # self.netM2.eval()
        self.pwc_model.eval()

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A_encoder)
        networks.print_network(self.netG_A_decoder)
        networks.print_network(self.netM)
        print('-----------------------------------------------')

    def set_input(self, input):
        print("set input")
        frames = input['frames']

        if len(self.gpu_ids) > 0:
            frames = frames.cuda(self.gpu_ids[0], async=True)

        self.frames = frames[0]
        self.image_paths = (input['A_paths'][0][0],input['A_paths'][1][0],input['A_paths'][2][0])
        self.index = input['index']

    def get_sa(self, img):
        # calc instance segmentation
        # input should be -1 ~ 1
        assert len(img.shape) == 3
        height, width = img.shape[1:]
        img = torch.stack((img[2], img[1], img[0]))
        inputs = {"image": (img+1)*127.5, "height": height, "width": width}
        y = self.NetIS([inputs])[0]["instances"]._fields
        mask = y["pred_masks"]
        classes = y["pred_classes"]
        cnt = 0
        nsa = torch.zeros_like(img[:1])
        nsa.unsqueeze(0)
        for t in range(classes.shape[0]):
            if classes[t] == 17:
                nsa = nsa + mask[t:t+1].float().unsqueeze(0) * (2 ** cnt)
                cnt += 1
                assert cnt < 10
        cnt = torch.max(nsa)
        if cnt > 0.5:
            nsa = nsa / torch.max(nsa)
        box = y['pred_boxes'].tensor.long()
        if (box.shape[0] == 0):
            box = (-1,-1,-1,-1)
        else:
            box = (torch.min(box[:,0]),torch.min(box[:,1]),torch.max(box[:,2]),torch.max(box[:,3]))
        return nsa, box

    def netG_A(self, x, sa=None):
        if self.opt.saliency:
            assert sa is not None
        return self.netG_A_decoder(self.netG_A_encoder(x, sa), sa)

    def scale_to(self, a, b, c, d, base=4, maxstep=4):
        a -= a%base
        c += (base-(c%base)) % base
        b -= b%base
        d += (base-(d%base)) % base
        for _ in range(maxstep):
            if a > 0:
                a-=base
            if b > 0:
                b-=base
            if c < self.frames.shape[3]:
                c += base
            if d < self.frames.shape[2]:
                d += base
        return a, b, c, d

    def getF(self, indexA=0, indexB=None, rev=False):
        # extract feature map
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

                # tmp = torch.mean(F, dim=1)
                # tmp = util.tensor2im(tmp[0].cpu())
                # cv2.imwrite("input_feature.png", tmp)

                F_all = F.clone()
                pos = list()
                for i in range(indexA, indexB):
                    a, b, c, d = [x.item() for x in self.box[i]]
                    a, b, c, d = self.scale_to(a, b, c, d)
                    sa_horse = self.SA[i:i+1, :, b:d, a:c]

                    # tmp = util.tensor2im(self.SA[i].cpu())
                    # cv2.imwrite("mask.png", tmp)

                    frame_horse = frame[i-indexA:i+1-indexA, :, b:d, a:c]

                    # tmp = util.tensor2im(frame_horse[0].cpu())
                    # cv2.imwrite("object_small.png", tmp[:,:,-1::-1])

                    h, w = d-b, c-a
                    sc = min(self.frames.shape[2]/h, self.frames.shape[3]/w)
                    nh = h*sc
                    nw = w*sc
                    nh = math.ceil(nh/4)*4
                    nw = math.ceil(nw/4)*4
                    frame_horse = nn.functional.interpolate(
                        frame_horse, size=(nh, nw), mode="bilinear")

                    # tmp = util.tensor2im(frame_horse[0].cpu())
                    # cv2.imwrite("object.png", tmp[:,:,-1::-1])

                    sa_horse_scaled = nn.functional.interpolate(
                        sa_horse, size=(nh, nw), mode="nearest")
                    F_horse = self.netG_A_encoder(frame_horse, sa_horse_scaled)

                    # tmp = torch.mean(F_horse, dim=1)
                    # tmp = util.tensor2im(tmp[0].cpu())
                    # cv2.imwrite("object_feature_small.png", tmp)

                    aa = a//4
                    bb = b//4
                    F_horse_scaled = nn.functional.interpolate(
                        F_horse, size=(h//4, w//4), mode="bilinear")

                    # tmp = torch.mean(F_horse_scaled, dim=1)
                    # tmp = util.tensor2im(tmp[0].cpu())
                    # cv2.imwrite("object_feature.png", tmp)

                    cc = aa+F_horse_scaled.shape[3]
                    dd = bb+F_horse_scaled.shape[2]
                    assert cc == c//4 and dd == d//4

                    sa_horse = nn.functional.interpolate(
                        (sa_horse > 0.0001).float(), size=(dd-bb, cc-aa), mode="nearest")
                    F_all[i-indexA:i-indexA+1, :,
                          bb:dd, aa:cc] = (F_horse_scaled * sa_horse) + F_all[i-indexA:i-indexA+1, :, bb:dd, aa:cc] * (1-sa_horse)
                    
                    # tmp = torch.mean(F_all, dim=1)
                    # tmp = util.tensor2im(tmp[0].cpu())
                    # cv2.imwrite("fused_feature.png", tmp)

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

    def test(self):
        tmp = list()
        import matplotlib.pyplot as plt
        if self.opt.saliency:
            sa = list()
            box = list()
            for img in self.frames:
                a,b = self.get_sa(img)
                if torch.sum(a) < 0.001:
                    return False
                sa.append(a)
                box.append(torch.stack(b))
            self.SA = torch.cat(sa, dim=0)
            self.sa = sa[0][0].data.cpu()
            self.box = box
        print("start")
        ret = list()
        f = list()
        boxed = list()
        scor = list()
        liml = 1
        
        # stylization serially

        # for i in range(self.frames.shape[0]):
        #     if self.opt.inpaint:
        #         F, F_horse, SA_horse, pos = self.getF(i,i+1)
        #     else:
        #         F = self.getF(i,i+1)

        #     if len(f) > 0:
        #         num = min(liml, len(f))

        #         # for j in range(4):
        #         #     tmp = util.tensor2im(self.frames[i-3+j])
        #         #     cv2.imwrite(str(j)+'_frame.png', tmp[:,:,-1::-1])
                
        #         # for j in range(0,3):
        #         #     tmp = torch.mean(f[i-3+j], dim=1)
        #         #     tmp = util.tensor2im(tmp[0])
        #         #     cv2.imwrite(str(j)+'_feature.png', tmp)
                
        #         optF = PWCnet.estimate(self.pwc_model, self.frames[i:i+1].repeat(num,1,1,1), self.frames[i-num:i])
        #         toptF = optF.permute(0, 3, 1, 2)
        #         tmp = nn.functional.interpolate(
        #             toptF, size=(F.shape[2], F.shape[3]) , mode="bilinear")
        #         tmp[:, 0, :, :] = tmp[:, 0, :, :] * tmp.shape[3] / toptF.shape[3]
        #         tmp[:, 1, :, :] = tmp[:, 1, :, :] * tmp.shape[2] / toptF.shape[2]
        #         FoptF = tmp.permute(0, 2, 3, 1)

        #         # for j in range(0,3):
        #         #     hsv = np.zeros((self.frames.shape[2], self.frames.shape[3], 3), dtype=np.uint8)
        #         #     hsv[..., 1] = 255
        #         #     flow = optF[j].cpu().numpy()
        #         #     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        #         #     hsv[..., 0] = ang * 180 / np.pi / 2
        #         #     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        #         #     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #         #     cv2.imwrite(str(j)+'_flow.png', bgr)

        #         oF = util.warp(torch.cat(f[-num:], dim=0), FoptF)
                
        #         # for j in range(0,3):
        #         #     tmp = torch.mean(oF[j], dim=0)
        #         #     tmp = util.tensor2im(tmp)
        #         #     cv2.imwrite(str(j)+'_feature_warped.png', tmp)
                
        #         # score = torch.softmax(self.netM(oF-F), dim=0)
        #         score = torch.softmax(self.netM(torch.cat((oF,F.repeat(num, 1,1,1)), dim=1)), dim=0)
                
        #         # for j in range(0,3):
        #         #     tmp = util.tensor2im(score[j])
        #         #     cv2.imwrite(str(j)+'_w_hat.png', tmp)
                
        #         oF = torch.sum(oF * score, dim=0, keepdim=True)

        #         # tmp = torch.mean(oF, dim=1)
        #         # tmp = util.tensor2im(tmp[0])
        #         # cv2.imwrite('feature_reference_fused.png', tmp)

        #         # score = (torch.tanh(self.netM2(oF-F)) + 1) / 2
        #         score = (torch.tanh(self.netM2(torch.cat((oF,F), dim=1))) + 1) / 2

        #         # tmp = util.tensor2im(score[0])
        #         # cv2.imwrite('w.png', tmp)
                
        #         scor.append(util.tensor2im(torch.mean(score[0], dim=0)))
        #         # scor.append(util.tensor2im(torch.abs(self.frames[i:i+1]-util.warp(self.frames[i-1:i], optF))))
        #         F = score * oF + (1-score) * F

        #         # tmp = torch.mean(F, dim=1)
        #         # tmp = util.tensor2im(tmp[0])
        #         # cv2.imwrite('feature_fused.png', tmp)

        #         del optF, toptF, tmp, score, FoptF, oF

        #     savetof = F
        #     ink = self.netG_A_decoder(F, self.SA[i:i+1])
        #     ret.append(util.tensor2im(ink))
        #     # if len(f) >= liml:
        #     #     for j in range(4):
        #     #         cv2.imwrite(str(j)+'_ink.png', ret[i-3+j][:,:,-1::-1])
        #     #     input()
        #     if len(f) >= liml:
        #         del f[0]
        #     f.append(savetof)

        #     if self.opt.inpaint:
        #         a,b,c,d = pos[0]

        #         ink_horse = self.netG_A_decoder(F_horse[0], SA_horse[0])
        #         tmp = util.tensor2im(ink_horse)
        #         tmp = cv2.resize(tmp, (c-a, d-b))
        #         del ink_horse
        #         ink_horse = tmp
        #         y, x = ink.shape[2:]
        #         ink_horse = np.pad(ink_horse, ((b, y-d), (a, x-c), (0, 0)), "constant", constant_values=255)
        #         boxed.append(ink_horse)

        #     del ink, F
        # return np.stack(ret), np.stack(boxed) if self.opt.inpaint else None, np.stack(scor) if len(scor) > 0 else None

        # stylization with frame reordering
        print("start ink")
        inked = set()
        uninked = set(range(self.frames.shape[0]))
        allFopt = dict()
        newf = [None] * self.frames.shape[0]

        # calc distance between nearest 50 frames
        maxl = 50
        # max number of reference frames
        maxr = 7
        def calc_dis(i, j, A, B):
            if abs(i-j) > maxl:
                return 1e10
            optF = PWCnet.estimate(self.pwc_model, A, B)
            toptF = optF.permute(0, 3, 1, 2)
            tmp = nn.functional.interpolate(
                toptF, size=(self.frames.shape[2]//4, self.frames.shape[3]//4) , mode="bilinear")
            tmp[:, 0, :, :] = tmp[:, 0, :, :] * tmp.shape[3] / toptF.shape[3]
            tmp[:, 1, :, :] = tmp[:, 1, :, :] * tmp.shape[2] / toptF.shape[2]
            FoptF = tmp.permute(0, 2, 3, 1)
            allFopt[(i, j)] = FoptF
            d = torch.mean(torch.sqrt(L2dis(optF)))
            del FoptF, tmp, optF, toptF
            return d

        dis = dict()
        for i in range(self.frames.shape[0]):
            print("calc optical flow {}/{}".format(i, self.frames.shape[0]))
            for j in range(i+1, self.frames.shape[0]):
                dis[(i, j)] = calc_dis(i, j, self.frames[i:i+1], self.frames[j:j+1])
                dis[(j, i)] = calc_dis(j, i, self.frames[j:j+1], self.frames[i:i+1])

        def evalue(x):
            val1 = 0
            for i in inked:
                val1 += dis[(x, i)]
            val2 = 0
            for i in uninked:
                if i != x:
                    val2 += dis[(x, i)]
            return - val1 / (len(inked)+1) + self.alpha * val2 / (len(uninked)+1)
        while len(uninked) > 0:
            idx = 0
            mn = 1e30
            for i in uninked:
                tmp = evalue(i)
                if tmp < mn:
                    mn = tmp
                    idx = i
            print("deal {} ...".format(idx))
            l = list()
            for i in range(self.frames.shape[0]):
                if i != idx and i in inked and abs(i-idx) <= maxl:
                    l.append((dis[(i, idx)], i))
            l.sort()
            if len(l) > maxr:
                l = l[:maxr]
            now = list(map(lambda x: newf[x[1]].cuda(), l))
            F, _, _, _ = self.getF(idx, idx+1)
            print("referrence:", l)

            if len(now) > 0:
                now = torch.cat(now, dim=0)
                FoptF = list()
                for i in range(len(now)):
                    FoptF.append(allFopt[(idx, l[i][1])])
                FoptF = torch.cat(FoptF, dim=0)
                now = util.warp(now, FoptF)
                score = torch.softmax(self.netM(now-F), dim=0)
                oF = torch.sum(now * score, dim=0, keepdim=True)
                score = (torch.tanh(self.netM2(oF-F)) + 1) / 2
                oF = oF * score + F * (1-score)
                del score, FoptF, now
            else:
                oF = F
            del F
            newf[idx] = oF.cpu()
            del oF
            inked.add(idx)
            uninked.remove(idx)
        del dis
        del allFopt
        del calc_dis
        del evalue
        fin = list()
        for i in range(len(newf)):
            fin.append(util.tensor2im(self.netG_A_decoder(newf[i].cuda(), self.SA[i:i+1])))
        del newf
        return np.stack(fin), None, None

    # get image paths
    def get_image_paths(self):
        return self.image_paths
