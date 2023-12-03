#!/usr/bin/python3

import argparse
import itertools

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR, Logger, ReplayBuffer
from .utils import weights_init_normal, get_config
from .datasets import ImageDataset, ValDataset
from reg_Model.CycleGan import *
from .utils import Resize, ToTensor, smooothing_loss, LCC, Bend_Penalty, GCC
from .utils import Logger, tensor2image
from .reg import Reg
from torchvision.transforms import RandomAffine
from torchvision.transforms import RandomAffine, ToPILImage
from .transformer import Transformer_2D
from sklearn import metrics
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import cv2
from skimage.metrics import mean_squared_error as compare_mse
class Cyc_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks 定义网络
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()  # 调用model.CycleGan函数
        self.netD_B = Discriminator(config['input_nc']).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=0.0005, betas=(0.5, 0.999))  # 鉴别器的优化器

        self.R_A = Reg(config['size'], config['size'], config['input_nc'], config['input_nc']).cuda()  # 定义配准网络
        self.spatial_transform = Transformer_2D().cuda()
        self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))


        if config['CycleGAN']:  # 是否使用CycleGan
            self.netG_B2A = Generator(config['input_nc'], config['output_nc']).cuda()
            self.netD_A = Discriminator(config['input_nc']).cuda()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                                lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation  输入和输出的内存分配
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])  # 1 1 256 256
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(1, 1).fill_(1.0), requires_grad=False) #all 1
        self.target_fake = Variable(Tensor(1, 1).fill_(0.0), requires_grad=False) #all 0

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Dataset loader
        level = config['noise_level']  # set noise level
        transforms_1 = [ToPILImage(),
                        RandomAffine(degrees=level, translate=[0.02 * level, 0.02 * level],
                                     scale=[1 - 0.02 * level, 1 + 0.02 * level], fill=-1),
                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]

        transforms_2 = [ToPILImage(),
                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]

        self.dataloader = DataLoader(
            ImageDataset(config['dataroot'], transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False),
            batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

        val_transforms = [ToTensor(),
                          Resize(size_tuple=(config['size'], config['size']))]

        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_=val_transforms, unaligned=False),
                                   batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

        # Loss plot
        self.logger = Logger(config['name'], config['port'], config['n_epochs'], len(self.dataloader))

    def train(self):
        global allMI
        global maxMI
        allMI = 0
        maxMI = 0
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))  # batchSize,input_nc,size,size MRI
                real_B = Variable(self.input_B.copy_(batch['B']))   # PAT
                if self.config['CycleGAN']:  # CycleGAN
                    if self.config['TranPAT']:  # CycleGAN + TranPAT
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                        Trans = self.R_A(real_A, fake_A)
                        SysRegist_A2B = self.spatial_transform(real_A, Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, fake_A)  ###SR
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + SR_loss + SM_loss
                        loss_Total.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################


                    else:  # CycleGAN + TranMRI
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                        Trans = self.R_A(fake_B, real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, real_B)  ###SR
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + SR_loss + SM_loss
                        loss_Total.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################



                else:  # GAN
                    if self.config['TranPAT']:  # GAN+TranPAT
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss
                        fake_A = self.netG_A2B(real_B)

                        Trans = self.R_A(real_A, fake_A)
                        SysRegist_A2B = self.spatial_transform(real_A, Trans)

                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, fake_A)  ###SR

                        pred_fake0 = self.netD_B(fake_A)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_real)
                        # LCC1_loss = self.config['LCC_lamda'] * LCC(SysRegist_A2B, fake_A)
                        # GCC1_loss = self.config['GCC_lamda'] * GCC(SysRegist_A2B, fake_A)

                        ####smooth loss
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        Bend1_loss = self.config['Bend_lamda'] * Bend_Penalty(Trans)

                        toal_loss = SM_loss + adv_loss +  SR_loss + Bend1_loss
                        toal_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_A = self.netG_A2B(real_B)
                        pred_fake0 = self.netD_B(fake_A)
                        pred_real = self.netD_B(real_A)
                        loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_fake) + self.config[
                            'Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        loss_D_B.backward()
                        self.optimizer_D_B.step()



                    else:  # # GAN+TranMRI
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss
                        fake_B = self.netG_A2B(real_A)

                        Trans = self.R_A(fake_B, real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, real_B)  ###SR

                        pred_fake0 = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_real)
                        # LCC1_loss = self.config['LCC_lamda'] * LCC(SysRegist_A2B, real_B)
                        # GCC1_loss = self.config['GCC_lamda'] * GCC(SysRegist_A2B, real_B)

                        ####smooth loss
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        # Bend1_loss = self.config['Bend_lamda'] * Bend_Penalty(Trans)

                        toal_loss = SM_loss + adv_loss + SR_loss
                        toal_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_B = self.netG_A2B(real_A)
                        pred_fake0 = self.netD_B(fake_B)
                        pred_real = self.netD_B(real_B)
                        loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_fake) + self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        ###################################

            # Save models checkpoints
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])

            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'TM/' + '{}netG_A2BTM.pth'.format(epoch))
            torch.save(self.R_A.state_dict(), self.config['save_root'] + 'TM/' + '{}RegistTM.pth'.format(epoch))




    def test(self, ):

        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2BOurs.pth'))
        self.R_A.load_state_dict(torch.load(self.config['save_root'] + 'RegistOurs.pth'))

        with torch.no_grad():
            InMAE = 0
            InPSNR = 0
            InSSIM = 0
            MAE = 0
            PSNR = 0
            SSIM = 0
            MSE = 0
            MI = 0
            NMI = 0
            CC = 0
            num = 0
            for i, batch in enumerate(self.val_data):

                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                fake_A = self.netG_A2B(real_B)
                Trans = self.R_A(real_A, fake_A)
                SysRegist_A2B = self.spatial_transform(real_A, Trans)
                SysRegist_A2B = SysRegist_A2B.detach().cpu().numpy().squeeze()

                num += 1

                plt.imsave("/public2/zhongyutian/RSAF-GAN/reg_result/num{}realC.jpg".format(num), SysRegist_A2B, cmap="gray")


    def Eva(self, ):  # Remember to modify the input location!!!
            MAE = 0
            PSNR = 0
            SSIM = 0
            MSE = 0
            MI = 0
            NMI = 0
            CC = 0
            num = 0
            for i, batch in enumerate(self.val_data):
                ###TranPAT
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))

                real_A = real_A.detach().cpu().numpy().squeeze()
                real_B = real_B.detach().cpu().numpy().squeeze()
                # real_B = (real_B - real_B.min()) / (real_B.max() - real_B.min()) * 2 - 1
                # real_A = (real_A - real_A.min()) / (real_A.max() - real_A.min()) * 2 - 1
                #
                mae = self.MAE(real_B, real_A)
                psnr = compare_psnr(real_B, real_A)
                ssim = compare_ssim(real_B, real_A)
                mse = compare_mse(real_B, real_A)
                mi = self.MI(real_B, real_A)
                nmi = self.NMI(real_B, real_A)
                cc = self.CC(real_B, real_A)

                #
                MAE += mae
                PSNR += psnr
                SSIM += ssim
                MSE += mse
                MI += mi
                NMI += nmi
                CC += cc
                num += 1
            print('real_A:', real_A)
            print('MAE:', MAE / num)
            print('MSE:', MSE / num)
            print('PSNR:', PSNR / num)
            print('SSIM:', SSIM / num)
            print('MI:', MI / num)
            print('NMI:', NMI / num)
            print('CC:', CC / num)

    def PSNR(self, fake, real):
        # x, y = np.where(real != -1)  # Exclude background
        mse = np.mean(((fake + 1) / 2. - (real + 1) / 2.) ** 2)
        if mse < 1.0e-10:
            return 100
        else:
            PIXEL_MAX = 1
            return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


    def MAE(self, fake, real):
        # x = np.where(real != -1)  # Exclude background
        # y = np.where(real != -1)
        mae = np.abs(fake - real).mean()
        return mae / 2  # from (-1,1) normaliz  to (0,1)

    def MI(self, fake, real):
        img_ref = fake.reshape(-1)
        img_sen = real.reshape(-1)
        size = img_ref.shape[-1]
        px = np.histogram(img_ref, 256, (0, 255))[0] / size
        py = np.histogram(img_sen, 256, (0, 255))[0] / size
        hx = - np.sum(px * np.log(px + 1e-8))
        hy = - np.sum(py * np.log(py + 1e-8))

        hxy = np.histogram2d(img_ref, img_sen, 256, [[0, 255], [0, 255]])[0]
        hxy /= (1.0 * size)
        hxy = - np.sum(hxy * np.log(hxy + 1e-8))

        r = hx + hy - hxy
        return r

    def NMI(self, fake, real):
        fake = np.reshape(fake, -1)
        real = np.reshape(real, -1)
        NMI = metrics.normalized_mutual_info_score(fake, real)
        return NMI

    def CC(self, fake, real):
        fake = np.reshape(fake, -1)
        real = np.reshape(real, -1)
        CC = np.corrcoef(fake, real)
        return CC[0, 1]

    def save_deformation(self, defms, root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max, x_min = dir_x.max(), dir_x.min()
        y_max, y_min = dir_y.max(), dir_y.min()
        dir_x = ((dir_x - x_min) / (x_max - x_min)) * 255
        dir_y = ((dir_y - y_min) / (y_max - y_min)) * 255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # tans_x[tans_x <= 150] = 0
        tans_x = cv2.applyColorMap(tans_x, COLORMAP_BONE)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # tans_y[tans_y <= 150] = 0
        tans_y = cv2.applyColorMap(tans_y, COLORMAP_BONE)
        gradxy = cv2.addWeighted(tans_x, 0.5, tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy)
