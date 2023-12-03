import random
import time
import datetime
import sys

import cv2
import matplotlib.pyplot as plt
import yaml
from torch.autograd import Variable
import torch
from visdom import Visdom
import torch.nn.functional as F
import numpy as np
class Resize():
    def __init__(self, size_tuple, use_cv = True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv


    def __call__(self, tensor):
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """
        tensor = tensor.unsqueeze(0)
 
        tensor = F.interpolate(tensor, size = [self.size_tuple[0],self.size_tuple[1]])

        tensor = tensor.squeeze(0)
 
        return tensor#1, 64, 128, 128
class ToTensor():
    def __call__(self, tensor):
        tensor = np.expand_dims(tensor, 0)
        return torch.from_numpy(tensor.copy())

def tensor2image(tensor):
    image = (127.5*(tensor.cpu().float().numpy()))+127.5
    image1 = image[0]
    for i in range(1,tensor.shape[0]):
        image1 = np.hstack((image1,image[i]))
    
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    #print ('image1.shape:',image1.shape)
    return image1.astype(np.uint8)


class Logger():
    def __init__(self, env_name ,ports, n_epochs, batches_epoch):
        #self.viz = Visdom(port= ports,env = env_name)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # Draw images
        # for image_name, tensor in images.items():
        #     if image_name not in self.image_windows:
        #         self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title': image_name})
        #         #print(image_name)
        #         plt.plot(tensor2image(tensor.data)[0,:,:])
        #         #plt.imsave("/public2/zhongyutian/Reg-GAN-main/data/output/Cyc/NC+R/img/1.jpg" , tensor2image(tensor.data)[0,:,:])
        #         #cv2.imwrite = ("/public2/zhongyutian/Reg-GAN-main/data/output/Cyc/NC+R/img/" + "epoch" + str(self.epoch) + "i" + str(self.batch) + "real_A.png", tensor2image(tensor.data)[0,:,:])
        #     else:
        #         self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name],
        #                        opts={'title': image_name})
        #         #print(tensor2image(tensor.data)[0,:,:].shape)
        #         #print(tensor2image(tensor.data).shape)
        #         #plt.imsave("/public2/zhongyutian/Reg-GAN-main/data/output/Cyc/NC+R/img/1.jpg","/public2/zhongyutian/Reg-GAN-main/data/output/Cyc/NC+R/img/2.jpg","/public2/zhongyutian/Reg-GAN-main/data/output/Cyc/NC+R/img/3.jpg",tensor2image(tensor.data)[0, :, :],cmap="gray")
        #         #cv2.imwrite = ("/public2/zhongyutian/Reg-GAN-main/data/output/Cyc/NC+R/img/" + "epoch" + str(self.epoch) + "i" + str(self.batch) + "real_A.png", tensor2image(tensor.data))

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            # for loss_name, loss in self.losses.items():
            #     if loss_name not in self.loss_windows:
            #         self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
            #                                                      Y=np.array([loss / self.batch]),
            #                                                      opts={'xlabel': 'epochs', 'ylabel': loss_name,
            #                                                            'title': loss_name})
            #     else:
            #         self.viz.line(X=np.array([self.epoch]), Y=np.array([loss / self.batch]),
            #                       win=self.loss_windows[loss_name], update='append')
            #     # Reset losses for next epoch
            #     self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    # print ('m:',m)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
        
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx*dx
    dy = dy*dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d 
    return d

def LCC(I, J):
    I2 = I.pow(2)
    J2 = J.pow(2)
    IJ = I * J
    win = [4, 4]
    eps = 1e-5
    filters = Variable(torch.ones(1, 1, win[0], win[1]))
    if I.is_cuda:  # gpu
        filters = filters.cuda()
    padding = (win[0] // 2, win[1] // 2)

    I_sum = F.conv2d(I, filters, stride=1, padding=padding)
    J_sum = F.conv2d(J, filters, stride=1, padding=padding)
    I2_sum = F.conv2d(I2, filters, stride=1, padding=padding)
    J2_sum = F.conv2d(J2, filters, stride=1, padding=padding)
    IJ_sum = F.conv2d(IJ, filters, stride=1, padding=padding)

    win_size = win[0] * win[1]

    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + eps)  # np.finfo(float).eps
    lcc = -1.0 * torch.mean(cc) + 1
    return lcc

def GCC(I, J):
    I2 = I.pow(2)
    J2 = J.pow(2)
    IJ = I * J
    # average value
    I_ave, J_ave = I.mean(), J.mean()
    I2_ave, J2_ave = I2.mean(), J2.mean()
    IJ_ave = IJ.mean()

    cross = IJ_ave - I_ave * J_ave
    I_var = I2_ave - I_ave.pow(2)
    J_var = J2_ave - J_ave.pow(2)

    #        cc = cross*cross / (I_var*J_var + np.finfo(float).eps)#1e-5
    cc = cross / (I_var.sqrt() * J_var.sqrt() + np.finfo(float).eps)  # 1e-5

    return -1.0 * cc + 1

def Bend_Penalty(pred):
    Ty = _diffs(pred, dim=0)
    Tx = _diffs(pred, dim=1)
    Tyy = _diffs(Ty, dim=0)
    Txx = _diffs(Tx, dim=1)
    Txy = _diffs(Tx, dim=0)
    p = Tyy.pow(2).mean() + Txx.pow(2).mean() + 2 * Txy.pow(2).mean()

    return p

def _diffs(y, dim):  # y shape(bs, nfeat, vol_shape)
    ndims = y.ndimension() - 2
    d = dim + 2
    # permute dimensions to put the ith dimension first
    #       r = [d, *range(d), *range(d + 1, ndims + 2)]
    y = y.permute(d, *range(d), *range(d + 1, ndims + 2))
    dfi = y[1:, ...] - y[:-1, ...]

    # permute back
    # note: this might not be necessary for this loss specifically,
    # since the results are just summed over anyway.
    #       r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
    df = dfi.permute(*range(1, d + 1), 0, *range(d + 1, ndims + 2))

    return df
