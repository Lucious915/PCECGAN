import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
from . import orcgan_networks
import sys
import pytorch_msssim


class ORCGANModel(BaseModel):
    def name(self):
        return 'ORCGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)
        self.input_A_U = self.Tensor(nb, 1, size, size)
        self.input_A_V = self.Tensor(nb, 1, size, size)
        self.input_B_gray = self.Tensor(nb, 1, size, size)


        if opt.vgg > 0:
            self.vgg_loss = orcgan_networks.PerceptualLoss(opt)
            if self.opt.IN_vgg:
                self.vgg_patch_loss = orcgan_networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg = orcgan_networks.load_vgg16("./model", self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif opt.fcn > 0:
            self.fcn_loss = orcgan_networks.SemanticLoss(opt)
            self.fcn_loss.cuda()
            self.fcn = orcgan_networks.load_fcn("./model")
            self.fcn.eval()
            for param in self.fcn.parameters():
                param.requires_grad = False
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        skip = True if opt.skip > 0 else False
        self.netG_A = orcgan_networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=skip, opt=opt)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
        #                                 opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=False, opt=opt)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = orcgan_networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, True, self.gpu_ids, False)
            if self.opt.patchD:
                self.netD_P = orcgan_networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_patchD, opt.norm, use_sigmoid, False, self.gpu_ids, True)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            print("which_epoch: " + str(which_epoch))
            self.load_network(self.netG_A, 'G_A', which_epoch)
            # self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.use_wgan:
                self.criterionGAN = orcgan_networks.DiscLossWGANGP()
            else:
                self.criterionGAN = orcgan_networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.use_mse:
                self.criterionCycle = torch.nn.MSELoss()
            else:
                self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))#, weight_decay=0.01
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))#, weight_decay=0.01
            if self.opt.patchD:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))#, weight_decay=0.01

        print('---------- Networks initialized -------------')
        orcgan_networks.print_network(self.netG_A)
        # networks.print_network(self.netG_B)
        if self.isTrain:
            orcgan_networks.print_network(self.netD_A)
            if self.opt.patchD:
                orcgan_networks.print_network(self.netD_P)
            # networks.print_network(self.netD_B)
        if opt.isTrain:
            self.netG_A.train()
            # self.netG_B.train()
        else:
            self.netG_A.eval()
            # self.netG_B.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_img = input['input_img']
        input_A_gray = input['A_gray']
        input_A_U = input['A_U']
        input_A_V = input['A_V']
        input_B_gray = input['B_gray']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_A_U.resize_(input_A_U.size()).copy_(input_A_U)
        self.input_A_V.resize_(input_A_V.size()).copy_(input_A_V)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_B_gray.resize_(input_B_gray.size()).copy_(input_B_gray)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        params=self.netG_A.state_dict() 

        # for k,v in params.items():
        #     print(k)
        # print("conv1_1=====================================================")
        # print(params['module.conv1_1.weight'])   
        # print(params['module.conv1_1.bias'])
        # print("bn1_1=======================================================")
        # print(params['module.bn1_1.weight'])  
        # print(params['module.bn1_1.bias'])
        # print("running_mean & running_var==================================")
        # print(params['module.bn1_1.running_mean'])   
        # print(params['module.bn1_1.running_var'])
        # print("conv1_2=====================================================")
        # print(params['module.conv1_2.weight'])   
        # print(params['module.conv1_2.bias'])
        # print("bn1_2=======================================================")
        # print(params['module.bn1_2.weight'])   
        # print(params['module.bn1_2.bias'])
        # print("running_mean & running_var==================================")
        # print(params['module.bn1_2.running_mean'])   
        # print(params['module.bn1_2.running_var'])


    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)
        # self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)


    def predict(self):
        


        self.real_A = Variable(self.input_A)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_A_U = Variable(self.input_A_U)
        self.real_A_V = Variable(self.input_A_V)

        self.Thr = random.uniform(0.01,0.05) #0~1

        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A, gray_out = self.netG_A.forward(self.real_A, self.real_A_gray, self.real_A_U, self.real_A_V, self.Thr)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray, self.real_A_constraint)
        # self.rec_A = self.netG_B.forward(self.fake_B)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        A_gray = util.atten2im(gray_out.data)
        latent_V, latent_H, latent_S = util.latent_dff_2im(self.latent_real_A.data)
        # rec_A = util.tensor2im(self.rec_A.data)
        # if self.opt.skip == 1:
        #     latent_real_A = util.tensor2im(self.latent_real_A.data)
        #     latent_show = util.latent2im(self.latent_real_A.data)
        #     max_image = util.max2im(self.fake_B.data, self.latent_real_A.data)
        #     return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
        #                     ('latent_show', latent_show), ('max_image', max_image), ('A_gray', A_gray)])
        # else:
        #     return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
        # return OrderedDict([('fake_B', fake_B)])
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_V', latent_V), ('latent_H', latent_H), ('latent_S', latent_S)])

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_A_basic(self, netD, real, fake, use_ragan):
        # Real
        m_fake = fake.detach()

        pred_real = netD.forward(real)
        pred_fake = netD.forward(m_fake)

        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, 
                                                real.data, fake.data)
        elif self.opt.use_ragan and use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5

        ## loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_A_basic(self.netD_A, self.real_B, fake_B, True)
        self.loss_D_A.backward()
        # for name, param in self.netD_A.named_parameters():
        #     print("D_A:"+name)
        #     print(param.grad)

    def backward_D_P_basic(self, netD, real, fake, use_ragan):
        # Real
        
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())

        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, 
                                                real.data, fake.data)
        elif self.opt.use_ragan and use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D

    
    
    def backward_D_P(self):
        if self.opt.hybrid_loss:
            loss_D_P = self.backward_D_P_basic(self.netD_P, self.real_patch, self.fake_patch, False)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_P_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], False)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        else:
            loss_D_P = self.backward_D_P_basic(self.netD_P, self.real_patch, self.fake_patch, True)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_P_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], True)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        if self.opt.D_P_times2:
            self.loss_D_P = self.loss_D_P*2

        self.loss_D_P.backward()
        # for name, param in self.netD_P.named_parameters():
        #     print("D_P:"+name)
        #     print(param.grad)

    def cal_batch_OR_loss(self, batch_img, threshold):
        batch_size = batch_img.size(0)
        r,g,b = batch_img[:,0,:,:]+1, batch_img[:,1,:,:]+1, batch_img[:,2,:,:]+1
        gray = (0.299*r+0.587*g+0.114*b)/2.
        gray = torch.unsqueeze(gray, 1)
        gray_thr = gray - threshold
        zero_ = torch.full_like(gray_thr, 0)
        batch_OR = torch.where(( gray_thr > 0 ), gray_thr, zero_)
        batch_OR_loss = batch_OR.contiguous().view(batch_size,-1).mean(1, keepdim=True)

        return batch_OR_loss

    def cal_batch_PC_loss(self, batch_fake_img, batch_real_img):
        # batch_size = batch_fake_img.size(0)
        # rf,gf,bf = batch_fake_img[:,0,:,:]+1, batch_fake_img[:,1,:,:]+1, batch_fake_img[:,2,:,:]+1
        # gray_fake = (0.299*rf+0.587*gf+0.114*bf)/2.
        # gray_fake = torch.unsqueeze(gray_fake, 1)
        # rr,gr,br = batch_real_img[:,0,:,:]+1, batch_real_img[:,1,:,:]+1, batch_real_img[:,2,:,:]+1
        # gray_real = (0.299*rr+0.587*gr+0.114*br)/2.
        # gray_real = torch.unsqueeze(gray_real, 1)
        # gray_thr = gray_fake - gray_real
        # zero_ = torch.full_like(gray_thr, 0)
        # batch_PC = torch.where(( gray_thr > 0 ), gray_thr, zero_)
        # # batch_OR_loss = batch_OR.contiguous().view(batch_size,-1).mean(1, keepdim=True)

        batch_fake_img_nor = (batch_fake_img+1)/2
        r,g,b = batch_fake_img_nor[0], batch_fake_img_nor[1], batch_fake_img_nor[2]
        fake_v = torch.max(torch.max(r,g),b)

        batch_real_img_nor = (batch_real_img+1)/2
        r,g,b = batch_real_img_nor[0], batch_real_img_nor[1], batch_real_img_nor[2]
        real_v = torch.max(torch.max(r,g),b)

        batch_PC = torch.pow(fake_v - real_v/2 , 2)

        return torch.mean(batch_PC)

    def cal_batch_Darkchannel_loss(self, batch_fake_img):
        batch_size = batch_fake_img.size(0)
        rf,gf,bf = batch_fake_img[:,0,:,:]+1, batch_fake_img[:,1,:,:]+1, batch_fake_img[:,2,:,:]+1
        gray_fake = (0.299*rf+0.587*gf+0.114*bf)/2.
        
        batch_DC_loss = gray_fake.contiguous().view(batch_size,-1).min(1, keepdim=True)[0]

        return batch_DC_loss

    def cal_batch_color_loss(self, batch_fake_img, batch_real_img):
        batch_size = batch_fake_img.size(0)
        # rf,gf,bf = batch_fake_img[:,0,:,:]+1, batch_fake_img[:,1,:,:]+1, batch_fake_img[:,2,:,:]+1
        # u_fake = (-0.169*rf-0.331*gf+0.5*bf)/2. + 0.5
        # u_fake = torch.unsqueeze(u_fake, 1)
        # v_fake = (0.5*rf-0.419*gf-0.081*bf)/2. + 0.5
        # v_fake = torch.unsqueeze(v_fake, 1)


        # rr,gr,br = batch_real_img[:,0,:,:]+1, batch_real_img[:,1,:,:]+1, batch_real_img[:,2,:,:]+1
        # u_real = (-0.169*rr-0.331*gr+0.5*br)/2. + 0.5
        # u_real = torch.unsqueeze(u_real, 1)
        # v_real = (0.5*rr-0.419*gr-0.081*br)/2. + 0.5
        # v_real = torch.unsqueeze(v_real, 1)

        # batch_color_loss = 1 - pytorch_msssim.ssim(u_real, u_fake) + 1 - pytorch_msssim.ssim(v_real, v_fake) 
        

        fake_img_nor = (batch_fake_img+1)/2
        r,g,b = fake_img_nor[:,0,:,:], fake_img_nor[:,1,:,:], fake_img_nor[:,2,:,:]

        mx = torch.max(torch.max(r,g),b)
        mn = torch.min(torch.min(r,g),b)
        df = mx-mn
        F_h = mx
        F_h = torch.where(mx==r,((60 * ((g-b)/df) + 360) % 360)/360,F_h)
        F_h = torch.where(mx==g,((60 * ((b-r)/df) + 120) % 360)/360,F_h)
        F_h = torch.where(mx==b,((60 * ((r-g)/df) + 240) % 360)/360,F_h)
        F_h = torch.where(mx==mn,torch.full_like(F_h, 0),F_h)
        F_h = torch.unsqueeze(F_h, 0)

        F_s = torch.where(mx==0,torch.full_like(df, 0),(df/mx))
        F_s = torch.unsqueeze(F_s, 0)


        real_img_nor = (batch_real_img+1)/2
        r,g,b = real_img_nor[:,0,:,:], real_img_nor[:,1,:,:], real_img_nor[:,2,:,:]

        mx = torch.max(torch.max(r,g),b)
        mn = torch.min(torch.min(r,g),b)
        df = mx-mn
        R_h = mx
        R_h = torch.where(mx==r,((60 * ((g-b)/df) + 360) % 360)/360,R_h)
        R_h = torch.where(mx==g,((60 * ((b-r)/df) + 120) % 360)/360,R_h)
        R_h = torch.where(mx==b,((60 * ((r-g)/df) + 240) % 360)/360,R_h)
        R_h = torch.where(mx==mn,torch.full_like(R_h, 0),R_h)
        R_h = torch.unsqueeze(R_h, 0)
        
        R_s = torch.where(mx==0,torch.full_like(df, 0),(df/mx))
        R_s = torch.unsqueeze(R_s, 0)

        

        
        batch_color_loss = 1 - pytorch_msssim.ssim(R_h, F_h) + 1 - pytorch_msssim.ssim(R_s, F_s) 
        

        # batch_color_loss = torch.abs(u_fake - u_real) + torch.abs(v_fake - v_real)
        return batch_color_loss#torch.mean(batch_color_loss)

    # def backward_D_B(self):
    #     fake_A = self.fake_A_pool.query(self.fake_A)
    #     self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    def forward(self,epoch):
        self.Thr = random.uniform(0.02,0.05) #0~1
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_A_U = Variable(self.input_A_U)
        self.real_A_V = Variable(self.input_A_V)
        self.real_B_gray = Variable(self.input_B_gray)

        self.real_img = Variable(self.input_img)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A, self.real_A_gray \
             = self.netG_A.forward(self.real_A, self.real_A_gray, self.real_A_U, self.real_A_V, self.Thr)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray, self.real_B, self.real_B_gray)


        if self.opt.patchD:
            batch_size = self.real_A.size(0)
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))

            
            self.fake_patch = self.fake_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]

            self.real_patch = self.real_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]

            self.input_patch = self.real_A[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]

            
            r,g,b = self.fake_patch[:,0,:,:]+1, self.fake_patch[:,1,:,:]+1, self.fake_patch[:,2,:,:]+1
            gray = (0.299*r+0.587*g+0.114*b)/2.
            gray = torch.unsqueeze(gray, 1)

            self.fake_patch_mean = gray.contiguous().view(batch_size,-1).mean(1, keepdim=True)+0.0001
            self.fake_patch_std = gray.contiguous().view(batch_size,-1).std(1, keepdim=True)+0.0001

            

            r,g,b = self.real_patch[:,0,:,:]+1, self.real_patch[:,1,:,:]+1, self.real_patch[:,2,:,:]+1
            gray = (0.299*r+0.587*g+0.114*b)/2.
            gray = torch.unsqueeze(gray, 1)

            self.real_patch_mean = gray.contiguous().view(batch_size,-1).mean(1, keepdim=True)+0.0001
            self.real_patch_std = gray.contiguous().view(batch_size,-1).std(1, keepdim=True)+0.0001

        if self.opt.patchD_3 > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []

            w = self.real_A.size(3)
            h = self.real_A.size(2)
            for i in range(self.opt.patchD_3):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                fake_patch_i = self.fake_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize]
                real_patch_i = self.real_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize]
                input_patch_i = self.real_A[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize]
                self.fake_patch_1.append(fake_patch_i)
                self.real_patch_1.append(real_patch_i)
                self.input_patch_1.append(input_patch_i)

                # print("self.fake_patch_mean_1"+str(torch.max(self.fake_patch_mean_1)))
                # print("self.fake_patch_std_1"+str(torch.max(self.fake_patch_std_1)))

                # print("self.real_patch_mean_1"+str(torch.max(self.real_patch_mean_1)))
                # print("self.real_patch_std_1"+str(torch.max(self.real_patch_std_1)))

            # w_offset_2 = random.randint(0, max(0, w - self.opt.patchSize - 1))
            # h_offset_2 = random.randint(0, max(0, h - self.opt.patchSize - 1))
            # self.fake_patch_2 = self.fake_B[:,:, h_offset_2:h_offset_2 + self.opt.patchSize,
            #        w_offset_2:w_offset_2 + self.opt.patchSize]
            # self.real_patch_2 = self.real_B[:,:, h_offset_2:h_offset_2 + self.opt.patchSize,
            #        w_offset_2:w_offset_2 + self.opt.patchSize]
            # self.input_patch_2 = self.real_A[:,:, h_offset_2:h_offset_2 + self.opt.patchSize,
            #        w_offset_2:w_offset_2 + self.opt.patchSize]

    def backward_G(self, epoch):

        m_fakeB = self.fake_B.detach()
        batch_size, c, h, w = m_fakeB.size()
        r,g,b = m_fakeB[:,0,:,:]+1, m_fakeB[:,1,:,:]+1, m_fakeB[:,2,:,:]+1
        # fake_gray = (0.299*r+0.587*g+0.114*b)/2.
        # fake_gray = torch.unsqueeze(fake_gray, 1)
        #-----------------------------------------------------------------------------------------
        # StyleAB_mean = m_fakeB.contiguous().view(batch_size,-1).mean(1, keepdim=True)+0.0001
        # StyleAB_std = m_fakeB.contiguous().view(batch_size,-1).std(1, keepdim=True)+0.0001
        #-----------------------------------------------------------------------------------------
        # loss_G_style_B = (self.StyleB_mean - StyleAB_mean)**2 + (self.StyleB_std - StyleAB_std)**2
        # loss_G_style_B = loss_G_style_B.mean()

        # loss_G_content_B = (self.ContentA - self.ContentAB)**2
        # loss_G_content_B = loss_G_content_B.mean()
        #-----------------------------------------------------------------------------------------
        # loss_G_fakeB_orc = 0.
        # for bx in range(0,batch_size):
        #     fakeB_r,fakeB_g,fakeB_b = m_fakeB[bx,0]+1, m_fakeB[bx,1]+1, m_fakeB[bx,2]+1
        #     fakeB_gray = (0.299*fakeB_r+0.587*fakeB_g+0.114*fakeB_b)/2.
        #     mh, mw = fakeB_gray.size()
        #     pixel_amt = 0.
        #     or_pix_amt = 0.
        #     for mhx in range(0,mh):
        #         for mwx in range(0,mw):
        #             pixel_amt += 1.
        #             if fakeB_gray[mhx,mwx] > self.Thr:
        #                 or_pix_amt += (fakeB_gray[mhx,mwx]-self.Thr)
                    
        #     loss_G_fakeB_orc += or_pix_amt/pixel_amt
        # loss_G_fakeB_orc = self.cal_batch_OR_loss(m_fakeB,self.Thr)  
        loss_power_constraint = self.cal_batch_PC_loss(m_fakeB,self.real_A)
        loss_color = self.cal_batch_color_loss(m_fakeB,self.real_A)
        loss_dark_channel = self.cal_batch_Darkchannel_loss(m_fakeB)
        

        
        pred_fakeB = self.netD_A.forward(self.fake_B)
        if self.opt.use_wgan:
            loss_G_fakeB_disc = -pred_fakeB.mean()
        elif self.opt.use_ragan:
            pred_realB = self.netD_A.forward(self.real_B)

            loss_G_fakeB_disc = (self.criterionGAN(pred_realB - torch.mean(pred_fakeB), False) +
                                      self.criterionGAN(pred_fakeB - torch.mean(pred_realB), True)) / 2
            
        else:
            loss_G_fakeB_disc = self.criterionGAN(pred_fakeB, True)

        
        self.loss_G_A = loss_G_fakeB_disc#loss_G_fakeB_orc.mean()*10 #+ loss_G_style_B + loss_G_content_B #
        

        loss_G_A = 0
        if self.opt.patchD:
            pred_fake_patch = self.netD_P.forward(self.fake_patch)
            if self.opt.hybrid_loss:
                loss_G_A += self.criterionGAN(pred_fake_patch, True)
            else:
                pred_real_patch = self.netD_P.forward(self.real_patch)
                
                loss_G_A += (self.criterionGAN(pred_real_patch - torch.mean(pred_fake_patch), False) +
                                      self.criterionGAN(pred_fake_patch - torch.mean(pred_real_patch), True)) / 2
        if self.opt.patchD_3 > 0:   
            for i in range(self.opt.patchD_3):
                pred_fake_patch_1 = self.netD_P.forward(self.fake_patch_1[i])
                if self.opt.hybrid_loss:
                    loss_G_A += self.criterionGAN(pred_fake_patch_1, True)
                else:
                    pred_real_patch_1 = self.netD_P.forward(self.real_patch_1[i])
                    
                    loss_G_A += (self.criterionGAN(pred_real_patch_1 - torch.mean(pred_fake_patch_1), False) +
                                        self.criterionGAN(pred_fake_patch_1 - torch.mean(pred_real_patch_1), True)) / 2
                    
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)

            else:
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)*2
        else:
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A
            else:
                self.loss_G_A += loss_G_A*2
        

        if epoch < 0:
            vgg_w = 0
        else:
            vgg_w = 1
        if self.opt.vgg > 0:
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg, 
                    self.fake_B, self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0
            if self.opt.patch_vgg:
                if not self.opt.IN_vgg:
                    loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.input_patch) * self.opt.vgg
                else:
                    loss_vgg_patch = self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.input_patch) * self.opt.vgg
                if self.opt.patchD_3 > 0:
                    for i in range(self.opt.patchD_3):
                        if not self.opt.IN_vgg:
                            loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                        else:
                            loss_vgg_patch += self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                    self.loss_vgg_b += loss_vgg_patch/float(self.opt.patchD_3 + 1)
                else:
                    self.loss_vgg_b += loss_vgg_patch
            self.loss_G = self.loss_G_A + self.loss_vgg_b*vgg_w
        elif self.opt.fcn > 0:
            self.loss_fcn_b = self.fcn_loss.compute_fcn_loss(self.fcn, 
                    self.fake_B, self.real_A) * self.opt.fcn if self.opt.fcn > 0 else 0
            if self.opt.patchD:
                loss_fcn_patch = self.fcn_loss.compute_vgg_loss(self.fcn, 
                    self.fake_patch, self.input_patch) * self.opt.fcn
                if self.opt.patchD_3 > 0:
                    for i in range(self.opt.patchD_3):
                        loss_fcn_patch += self.fcn_loss.compute_vgg_loss(self.fcn, 
                            self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.fcn
                    self.loss_fcn_b += loss_fcn_patch/float(self.opt.patchD_3 + 1)
                else:
                    self.loss_fcn_b += loss_fcn_patch
            self.loss_G = self.loss_G_A + self.loss_fcn_b*vgg_w
            

        # self.loss_G = self.L1_AB + self.L1_BA
        
        # print("self.loss_G:"+str(self.loss_G))
        # Variable(self.loss_G, requires_grad = True)
        
        self.loss_G += Variable(loss_color.mean(), requires_grad = True) #loss_power_constraint.mean()*10 +
        # self.loss_G += Variable(loss_power_constraint.mean()*10, requires_grad = True)
        # self.loss_G += Variable(loss_dark_channel.mean(), requires_grad = True)
        self.loss_G.backward()
        # for name, param in self.netG_A.named_parameters():
        #     print("G_A:"+name)
        #     print("G_A.data:"+str(param.data))
        #     print("G_A.grad:"+str(param.grad))



    # def optimize_parameters(self, epoch):
    #     # forward
    #     self.forward()
    #     # G_A and G_B
    #     self.optimizer_G.zero_grad()
    #     self.backward_G(epoch)
    #     self.optimizer_G.step()
    #     # D_A
    #     self.optimizer_D_A.zero_grad()
    #     self.backward_D_A()
    #     self.optimizer_D_A.step()
    #     if self.opt.patchD:
    #         self.forward()
    #         self.optimizer_D_P.zero_grad()
    #         self.backward_D_P()
    #         self.optimizer_D_P.step()
        # D_B
        # self.optimizer_D_B.zero_grad()
        # self.backward_D_B()
        # self.optimizer_D_B.step()
    def optimize_parameters(self, epoch):
        # forward
        self.forward(epoch)
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        torch.nn.utils.clip_grad_value_(self.netG_A.parameters(), 1.)
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        if not self.opt.patchD:
            torch.nn.utils.clip_grad_value_(self.netD_A.parameters(), 1.)
            self.optimizer_D_A.step()
        else:
            # self.forward()
            self.optimizer_D_P.zero_grad()
            self.backward_D_P()
            torch.nn.utils.clip_grad_value_(self.netD_A.parameters(), 1.)
            self.optimizer_D_A.step()
            torch.nn.utils.clip_grad_value_(self.netD_P.parameters(), 1.)
            self.optimizer_D_P.step()


    def get_current_errors(self, epoch):
        D_A = self.loss_D_A.data[0]
        D_P = self.loss_D_P.data[0] if self.opt.patchD else 0
        G_A = self.loss_G_A.data[0]
        if self.opt.vgg > 0:
            vgg = self.loss_vgg_b.data[0]/self.opt.vgg if self.opt.vgg > 0 else 0
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ("vgg", vgg), ("D_P", D_P)])
        elif self.opt.fcn > 0:
            fcn = self.loss_fcn_b.data[0]/self.opt.fcn if self.opt.fcn > 0 else 0
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ("fcn", fcn), ("D_P", D_P)])
        

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        if self.opt.skip > 0:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
            latent_show = util.latent2im(self.latent_real_A.data)
            if self.opt.patchD:
                fake_patch = util.tensor2im(self.fake_patch.data)
                real_patch = util.tensor2im(self.real_patch.data)
                if self.opt.patch_vgg:
                    input_patch = util.tensor2im(self.input_patch.data)
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('input_patch', input_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('input_patch', input_patch), ('self_attention', self_attention)])
                else:
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('self_attention', self_attention)])
            else:
                if not self.opt.self_attention:
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B)])
                else:
                    self_attention = util.atten2im(self.real_A_gray.data)
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('latent_real_A', latent_real_A), ('latent_show', latent_show),
                                    ('self_attention', self_attention)])
        else:
            if not self.opt.self_attention:
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
            else:
                self_attention = util.atten2im(self.real_A_gray.data)
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('self_attention', self_attention)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        if self.opt.patchD:
            self.save_network(self.netD_P, 'D_P', label, self.gpu_ids)
        # self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        # self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):
        
        if self.opt.new_lr:
            lr = self.old_lr/2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        if self.opt.patchD:
            for param_group in self.optimizer_D_P.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
