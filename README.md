# InkVideoStyleTransfer

This is the official implemet of **Instance-Aware Coherent Video Style Transfer for Chinese Ink Wash Painting** (IJCAI-21).

## Envienment Requirements 

Pytorch >= 1.4

Detectron2

## Train & Eval
run train_up.sh or test_up.sh to train or evaluate this model.

Remember to modify the paths in these two files.

## Pretrained Models
You need to download [Hed](https://github.com/xwjabc/hed), PointRend model from [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/docs/index.rst) and [PWC-Net](https://github.com/sniklaus/pytorch-pwc).

Remember to modify the paths in models/cycle_gan_model.py and models/test_model.py

Our pretrained model is [here](https://1drv.ms/u/s!AtiCcSW9ZS9IqAm0wM-D3DPz2Bqf?e=DDuqsm).
