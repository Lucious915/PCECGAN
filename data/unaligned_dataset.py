import torch
from torch import nn
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, store_dataset
import random
from PIL import Image
import PIL
from pdb import set_trace as st

def pad_tensor(input):
    
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

            padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        # self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)
        self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)

        # self.A_paths = sorted(self.A_paths)
        # self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        print("self.A_size"+str(self.A_size))
        print("self.B_size"+str(self.B_size))
        
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        # A_path = self.A_paths[index % self.A_size]
        # B_path = self.B_paths[index % self.B_size]

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        A_img = self.A_imgs[index % self.A_size]
        B_img = self.B_imgs[index % self.B_size]
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        # A_size = A_img.size
        # B_size = B_img.size
        # A_size = A_size = (A_size[0]//16*16, A_size[1]//16*16)
        # B_size = B_size = (B_size[0]//16*16, B_size[1]//16*16)
        # A_img = A_img.resize(A_size, Image.BICUBIC)
        # B_img = B_img.resize(B_size, Image.BICUBIC)
        # A_gray = A_img.convert('LA')
        # A_gray = 255.0-A_gray

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        
        if self.opt.resize_or_crop == 'no':
            input_img = A_img
            # r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1

            # # r = torch.pow(r/2,1.7)*2
            # # g = torch.pow(g/2,1.7)*2
            # # b = torch.pow(b/2,1.7)*2

            # A_gray = (0.299*r+0.587*g+0.114*b)/2.
            # A_gray = torch.unsqueeze(A_gray, 0)

            # A_U = (-0.169*r-0.331*g+0.5*b)/2. + 0.5
            # A_U = torch.unsqueeze(A_U, 0)
            # # A_U = torch.where(A_U > 1, torch.full_like(A_U, 1), A_U)
            # # A_U = torch.where(A_U < 0, torch.full_like(A_U, 0), A_U)
            
            # A_V = (0.5*r-0.419*g-0.081*b)/2. + 0.5
            # A_V = torch.unsqueeze(A_V, 0)
            # # A_V = torch.where(A_V > 1, torch.full_like(A_V, 1), A_V)
            # # A_V = torch.where(A_V < 0, torch.full_like(A_V, 0), A_V)

            # r,g,b = B_img[0]+1, B_img[1]+1, B_img[2]+1
            # B_gray = (0.299*r+0.587*g+0.114*b)/2.
            # B_gray = torch.unsqueeze(B_gray, 0)
            input_img_nor = (input_img+1)/2
            # input_img_nor = torch.pow(input_img_nor,gamma)
            r,g,b = input_img_nor[0], input_img_nor[1], input_img_nor[2]

            mx = torch.max(torch.max(r,g),b)

            mn = torch.min(torch.min(r,g),b)

            df = mx-mn

            A_h = mx
            A_h = torch.where(mx==r,((60 * ((g-b)/df) + 360) % 360)/360,A_h)
            A_h = torch.where(mx==g,((60 * ((b-r)/df) + 120) % 360)/360,A_h)
            A_h = torch.where(mx==b,((60 * ((r-g)/df) + 240) % 360)/360,A_h)
            A_h = torch.where(mx==mn,torch.full_like(A_h, 0),A_h)
            A_h = torch.unsqueeze(A_h, 0)

            A_s = torch.where(mx==0,torch.full_like(df, 0),(df/mx))
            A_s = torch.unsqueeze(A_s, 0)

            A_v = mx
            A_v = torch.unsqueeze(A_v, 0)

            B_nor = (B_img+1)/2
            r,g,b = B_nor[0], B_nor[1], B_nor[2]
            B_v = torch.max(torch.max(r,g),b)
            B_v = torch.unsqueeze(B_v, 0)

            
        else:
            w = A_img.size(2)
            h = A_img.size(1)
            
            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
            # if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
            #     times = random.randint(self.opt.low_times,self.opt.high_times)/100.
            #     input_img = (A_img+1)/2./times
            #     input_img = input_img*2-1
            # else:
            #     input_img = A_img
            input_img = A_img
            if self.opt.lighten:
                B_img = (B_img + 1)/2.
                B_img = (B_img - torch.min(B_img))/(torch.max(B_img) - torch.min(B_img))
                B_img = B_img*2. -1


            # r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1

            # gamma = random.uniform(0.1,1)
            # r = torch.pow(r/2,gamma)*2
            # g = torch.pow(g/2,gamma)*2
            # b = torch.pow(b/2,gamma)*2
            # A_gray = (0.299*r+0.587*g+0.114*b)/2.
            # # A_gray /= A_gray.max()
            # # 
            
            
            # A_gray = torch.unsqueeze(A_gray, 0)
            # A_gray = torch.where(A_gray > 1, torch.full_like(A_gray, 1), A_gray)
            # A_gray = torch.where(A_gray < 0, torch.full_like(A_gray, 0), A_gray)
            
            # A_U = (-0.169*r-0.331*g+0.5*b)/2. + 0.5
            # A_U = torch.unsqueeze(A_U, 0)
            # A_U = torch.where(A_U > 1, torch.full_like(A_U, 1), A_U)
            # A_U = torch.where(A_U < 0, torch.full_like(A_U, 0), A_U)
            
            # A_V = (0.5*r-0.419*g-0.081*b)/2. + 0.5
            # A_V = torch.unsqueeze(A_V, 0)
            # A_V = torch.where(A_V > 1, torch.full_like(A_V, 1), A_V)
            # A_V = torch.where(A_V < 0, torch.full_like(A_V, 0), A_V)

            # r = A_gray[0] + (1.13983 * (A_V[0]-0.5))
            # g = A_gray[0] - (0.39465 * (A_U[0]-0.5)) - (0.58060 * (A_V[0]-0.5))
            # b = A_gray[0] + (2.03211 * (A_U[0]-0.5))
            # r = torch.unsqueeze(r, 0)
            # g = torch.unsqueeze(g, 0)
            # b = torch.unsqueeze(b, 0)
            # input_img = torch.cat((r, g), 0)
            # input_img = torch.cat((input_img, b), 0)
            # input_img = input_img*2 - 1
            # input_img = torch.where(input_img > 1, torch.full_like(input_img, 1), input_img)
            # input_img = torch.where(input_img < -1, torch.full_like(input_img, -1), input_img)
            # A_img = input_img

            

            # r,g,b = B_img[0]+1, B_img[1]+1, B_img[2]+1
            # B_gray = (0.299*r+0.587*g+0.114*b)/2.
            # B_gray = torch.unsqueeze(B_gray, 0)

            gamma = random.uniform(0.1,1)
            input_img_nor = (input_img+1)/2
            input_img_nor = torch.pow(input_img_nor,gamma)
            r,g,b = input_img_nor[0], input_img_nor[1], input_img_nor[2]

            mx = torch.max(torch.max(r,g),b)

            mn = torch.min(torch.min(r,g),b)

            df = mx-mn

            A_h = mx
            A_h = torch.where(mx==r,((60 * ((g-b)/df) + 360) % 360)/360,A_h)
            A_h = torch.where(mx==g,((60 * ((b-r)/df) + 120) % 360)/360,A_h)
            A_h = torch.where(mx==b,((60 * ((r-g)/df) + 240) % 360)/360,A_h)
            A_h = torch.where(mx==mn,torch.full_like(A_h, 0),A_h)
            A_h = torch.unsqueeze(A_h, 0)

            A_s = torch.where(mx==0,torch.full_like(df, 0),(df/mx))
            A_s = torch.unsqueeze(A_s, 0)

            A_v = mx
            A_v = torch.unsqueeze(A_v, 0)

            # gamma = random.uniform(1,2)
            B_nor = (B_img+1)/2
            B_nor = torch.pow(B_nor,gamma)
            r,g,b = B_nor[0], B_nor[1], B_nor[2]
            B_v = torch.max(torch.max(r,g),b)
            B_v = torch.unsqueeze(B_v, 0)
            

            
        return {'A': A_img, 'B': B_img, 'A_gray': A_v, 'A_U':A_h, 'A_V':A_s, 'B_gray': B_v, 'input_img': input_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'


