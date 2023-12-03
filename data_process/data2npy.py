# 尝试将转化的npy文件展示其中的内容。
import numpy as np
from PIL import Image
import os
import cv2
np.set_printoptions(threshold=np.inf)
# #####################  im 2 npy  #################

path = r'/mnt/dfc_data2/project/linyusen/zhongyutian/zhongyutian/RSAF-GAN/reg_data/1/B'
out_path = r'/mnt/dfc_data2/project/linyusen/zhongyutian/zhongyutian/RSAF-GAN/reg_data/1/B/'


def get_file(path, rule='.jpg'):
    all = []
    for fpathe, dirs, files in os.walk(path):  # all 目录
        for f in files:
            filename = os.path.join(fpathe, f)
            if filename.endswith(rule):
                all.append((filename))
    return all


def normalization(x):
    return (x-x.min())/(x.max()-x.min())*2-1

if __name__ == '__main__':
    paths = get_file(path, rule='.jpg')
    for ims in paths:
        #print(ims)
        # cut path and '.jpg' ,保留 000000050755_bicLRx4图片名称
        path = os.path.dirname(ims)
        file_name = ims[len(path)+1:]
        # file_name = ims.strip(r'/public2/zhongyutian/Reg-GAN-main/PATMRI/PAT/')
        file_name = file_name.strip('.jpg')
        # 可以打印出来看看cut以后的效果
        #print((file_name,type(file_name)))
        #ims="/public2/zhongyutian/Reg-GAN-main/001.jpg"
        im1 = cv2.imread(ims)
        im2 = np.array(im1[:,:,0],dtype=np.float64)
        im2 = normalization(im2)
        # print(type(im2))
        # print(im2.dtype)

        np.save(out_path+file_name + '.npy', im2)
        # np.save('/public2/zhongyutian/Reg-GAN-main/oral1.npy',im2)
        # a=np.load("/public2/zhongyutian/Reg-GAN-main/oral1.npy")
        # print(a.shape)
