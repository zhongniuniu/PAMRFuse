import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_loss(n):
    y = []
    for i in range(0,n):
        enc = np.load('C:/Users/Admin/Desktop/loss/MAE_epoch_{}.npy'.format(i))
        # enc = torch.load('D:\MobileNet_v1\plan1-AddsingleLayer\loss\epoch_{}'.format(i))
        tempy = list(enc)
        y += tempy
    x = range(0,len(y))
    plt.plot(x, y, '.-')
    plt_title = 'BATCH_SIZE = 32; LEARNING_RATE:0.001'
    plt.title(plt_title)
    plt.xlabel('per 200 times')
    plt.ylabel('LOSS')
    # plt.savefig(file_name)
    plt.show()

if __name__ == "__main__":
    plot_loss(39)