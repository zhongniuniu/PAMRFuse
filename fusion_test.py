from __future__ import print_function

import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
import h5py
import numpy as np
from fusion_model.generate import generate
import scipy.ndimage
import scipy.io as scio

BATCH_SIZE = 1
EPOCHES = 4
LOGGING = 20

MODEL_SAVE_PATH = '/public2/zhongyutian/RSAF-GAN/weights/fusion/99/'

def main():
	print('\nBegin to generate pictures ...\n')
	path = '/public2/zhongyutian/RSAF-GAN/fusion_data/OursregPATMRI/'
	Format='.jpg'

	T=[]
	for i in range(99):
		index = i + 1
		oe_path = path + 'u' + str(index) + Format
		ue_path = path + 'o' + str(index) + Format

		t = generate(oe_path, ue_path, MODEL_SAVE_PATH, index, output_path = '/public2/zhongyutian/RSAF-GAN/fusion_result/', format=Format)
		# t = color_generate(oe_path, ue_path, MODEL_SAVE_PATH, index, output_path='./fusion_imgOutput/color/', format=Format)

		T.append(t)
		print("%s time: %s" % (index, t))
	scio.savemat('time.mat', {'T': T})

if __name__ == '__main__':
	main()
