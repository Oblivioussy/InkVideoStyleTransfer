import os
for i in os.listdir("../test"):
    d = os.path.join("../test", i)
    if i.find("test") >= 0 and os.path.isdir(d) and i.find("test6") == -1:
        for j in os.listdir(d):
            d2 = os.path.join(d, j)
            if os.path.isdir(d2) and d2.find("_res") == -1 and os.path.isdir(d2):
                fl = True
                for k in os.listdir(d2):
                    d3 = os.path.join(d2, k)
                    if os.path.isdir(d3) and d3.find("_res") == -1:
                        fl = False
                        os.system("python2 test.py --dataroot {} --name horse2_cyclegan_edge_10_dec_150 --model cycle_gan --no_dropout  --how_many {}  --loadSize 512 --display_id 0 --resize_or_crop scale_width --results_dir {} --dataset_mode single --model test --saliency".format(d3, len(os.listdir(d3)), d3+"_res"))
                if fl:
                    os.system("python2 test.py --dataroot {} --name horse2_cyclegan_edge_10_dec_150 --model cycle_gan --no_dropout  --how_many {}  --loadSize 512 --display_id 0 --resize_or_crop scale_width --results_dir {} --dataset_mode single --model test --saliency".format(d2, len(os.listdir(d2)), d2+"_res"))