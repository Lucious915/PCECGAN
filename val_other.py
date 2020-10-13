# conda activate slic
# python3 val.py

import math
import glob
from PIL import Image
from SSIM_PIL import compare_ssim
import numpy as np
import numpy.matlib
import os

def logGabor_filter(rows,cols,omega0,sigmaF):
    cm, rm = np.meshgrid( (np.arange(cols)-np.fix(cols/2)) / (cols-np.mod(cols,2)) ,
        (np.arange(rows)-np.fix(rows/2)) / (rows-np.mod(rows,2)))
    mask_one = np.ones((rows,cols))
    mask_zero = np.zeros((rows,cols))
    mask = np.where( (cm**2+rm**2)>0.25 , mask_zero , mask_one)

    cm = np.multiply(cm,mask)
    rm = np.multiply(rm,mask)
    cm = np.fft.ifftshift(cm)
    rm = np.fft.ifftshift(rm)

    radius = np.sqrt(cm**2+rm**2)
    radius[0,0] = 1;

    LG = np.exp( -(np.log(radius/omega0)**2) / (2*(sigmaF**2)))
    LG[0,0] = 0

    return LG


def rgb2lab(inputRGB):
    #https://stackoverflow.com/questions/13405956/convert-an-image-rgb-lab-with-python
    num = 0
    RGB = [0, 0, 0]
    inputRGB_n = inputRGB/ 255.
    RGB = np.where( inputRGB_n>0.04045 , ((inputRGB_n+0.055)/1.055)**2.4 , inputRGB_n/12.92)
    RGB *= 100
    
    XYZ = RGB
    XYZ[:,:,0] = RGB [:,:,0] * 0.4124 + RGB [:,:,1] * 0.3576 + RGB [:,:,2] * 0.1805
    XYZ[:,:,1] = RGB [:,:,0] * 0.2126 + RGB [:,:,1] * 0.7152 + RGB [:,:,2] * 0.0722
    XYZ[:,:,2] = RGB [:,:,0] * 0.0193 + RGB [:,:,1] * 0.1192 + RGB [:,:,2] * 0.9505
    XYZ = np.round(XYZ,4)
    XYZ[:,:,0] = XYZ[:,:,0] / 95.047            # ref_X =  95.047   Observer= 2°, Illuminant= D65
    XYZ[:,:,1] = XYZ[:,:,1] / 100.0             # ref_Y = 100.000
    XYZ[:,:,2] = XYZ[:,:,2] / 108.883           # ref_Z = 108.883
    XYZ = np.where( XYZ > 0.008856 , XYZ**( 0.3333333333333333 ) , 7.787*XYZ + (16./116.) )
    
    LAB = XYZ
    LAB[:,:,0] = ( 116 * XYZ[:,:,1] ) - 16
    LAB[:,:,1] = 500 * ( XYZ[:,:,0] - XYZ[:,:,1] )
    LAB[:,:,2] = 200 * ( XYZ[:,:,1] - XYZ[:,:,2] )
    LAB = np.round(LAB,4)

    return LAB

def rgb2lmn(inputRGB):
    RGB = inputRGB/ 255.
    LMN = RGB
    LMN[:,:,0] = 0.06 * RGB[:,:,0] + 0.63 * RGB[:,:,1] + 0.27 * RGB[:,:,2]
    LMN[:,:,1] = 0.30 * RGB[:,:,0] + 0.04 * RGB[:,:,1] - 0.35 * RGB[:,:,2]
    LMN[:,:,2] = 0.34 * RGB[:,:,0] - 0.6  * RGB[:,:,1] + 0.17 * RGB[:,:,2]
    
    return LMN

def sdsp(img):
    
    sigmaF = 6.2
    omega0 = 0.002
    sigmaD = 114
    sigmaC = 0.25
    lab = rgb2lab(img)
    l_channel = lab[:,:,0]
    a_channel = lab[:,:,1]
    b_channel = lab[:,:,2]
    l_fft = np.fft.fft2(l_channel)
    a_fft = np.fft.fft2(a_channel)
    b_fft = np.fft.fft2(b_channel)
    rows, cols, channels = img.shape
    LG = logGabor_filter(rows,cols,omega0,sigmaF)

    Final_L = np.real(np.fft.ifft2(np.multiply(l_fft,LG)))
    Final_A = np.real(np.fft.ifft2(np.multiply(a_fft,LG)))
    Final_B = np.real(np.fft.ifft2(np.multiply(b_fft,LG)))

    SFMap = np.sqrt(Final_L**2+Final_A**2+Final_B**2)
    
    #the central areas will have a bias towards attention
    coordinateMtx = np.zeros((rows,cols,2))
    coordinateMtx[:,:,0] = np.matlib.repmat(np.arange(rows),cols,1).T
    coordinateMtx[:,:,1] = np.matlib.repmat(np.arange(cols),rows,1)
    
    centerMtx = np.ones((rows,cols,2))
    centerY = rows / 2
    centerX = cols / 2
    centerMtx[:,:,0] *= centerY
    centerMtx[:,:,1] *= centerX

    SDMap = np.exp(-np.sum((coordinateMtx - centerMtx)**2,2) / sigmaD**2)

    #warm colors have a bias towards attention
    maxA = np.max(a_channel)
    minA = np.min(a_channel)
    normalized_A = (a_channel-minA) / (maxA - minA)

    maxB = np.max(b_channel)
    minB = np.min(b_channel)
    normalized_B = (b_channel-minB) / (maxB - minB)

    lab_dist_square = normalized_A**2 + normalized_B**2
    SCMap = 1- np.exp(-lab_dist_square / (sigmaC**2))
    VSMap = np.multiply(np.multiply(SFMap,SDMap),SCMap)

    return VSMap

def scharr_gradient_filter(img):
    
    lmn = rgb2lmn(img)
    img_l = lmn[:,:,0]
    pad_img = np.pad(img_l, ((1,1) ,(1,1)) , 'edge')
    
    #img = pad_img[1:-1,1:-1]
    shift_l = pad_img[:-2   , 1:-1] #left
    shift_r = pad_img[2:    , 1:-1] #right
    shift_u = pad_img[1:-1  , :-2]  #up
    shift_d = pad_img[1:-1  , 2:]   #down

    shift_lu = pad_img[:-2  , :-2]
    shift_ru = pad_img[2:   , :-2]
    shift_ld = pad_img[:-2  , 2:]
    shift_rd = pad_img[2:   , :-2]

    SGX = (10*shift_l -10*shift_r +3*shift_lu +3*shift_ld -3*shift_ru -3*shift_rd) /16
    SGY = (10*shift_u -10*shift_d +3*shift_lu +3*shift_ru -3*shift_ld -3*shift_rd) /16

    SG = np.sqrt(SGX**2 + SGY**2)
    
    return SG

def compute_vsi(img1, img2):
    C1 = 1
    C2 = 1
    C3 = 1
    Alpha = 1
    Beta = 1
    buffer_1 = np.asarray(img1)
    buffer_2 = np.asarray(img2)

    VSMap1 = sdsp(buffer_1)
    VSMap2 = sdsp(buffer_2)
    VSm = np.maximum(VSMap1,VSMap2)
    Svs = (2*np.multiply(VSMap1,VSMap2)+C1) / (VSMap1**2+VSMap2**2+C1)

    GMap1 = scharr_gradient_filter(buffer_1)
    GMap2 = scharr_gradient_filter(buffer_2)
    Sg = (2*np.multiply(GMap1,GMap2)+C2) / (GMap1**2+GMap2**2+C2)

    LMN1 = rgb2lmn(buffer_1)
    LMN2 = rgb2lmn(buffer_2)
    SM = (2*np.multiply(LMN1[:,:,1],LMN2[:,:,1])+C3) / (LMN1[:,:,1]**2+LMN2[:,:,1]**2+C3)
    SN = (2*np.multiply(LMN1[:,:,2],LMN2[:,:,2])+C3) / (LMN1[:,:,2]**2+LMN2[:,:,2]**2+C3)
    Sc = np.multiply(SM,SN)

    Sgc = np.multiply(Sg**Alpha,Sc**Beta)
    S = np.multiply(Svs,Sgc)

    VSI = np.sum( np.multiply(S,VSm) ) / np.sum(VSm)

    return VSI

def compute_nde(img_enhance, img_origin):
    buffer_1 = np.asarray(img_origin)
    buffer_2 = np.asarray(img_enhance)

    lmn_1 = rgb2lmn(buffer_1)
    img_l1 = lmn_1[:,:,0]
    
    lmn_2 = rgb2lmn(buffer_2)
    img_l2 = lmn_2[:,:,0]

    hist_1 = np.zeros(256) + 0.0001
    hist_2 = np.zeros(256) + 0.0001
    
    for cols in img_l1:
        for p in cols:
            hist_1[int(p*255)] += 1
    hist_1 /= img_l1.size

    for cols in img_l2:
        for p in cols:
            hist_2[int(p*255)] += 1
    hist_2 /= img_l2.size

    discrete_entropy1 = -np.sum(np.multiply(hist_1,np.log(hist_1)))
    discrete_entropy2 = -np.sum(np.multiply(hist_2,np.log(hist_2)))

    nDE = 1 / ( 1 + ( (np.log(256)-discrete_entropy2) / (np.log(256)-discrete_entropy1) ) )

    return nDE

def compute_eme(img_enhance, img_origin):
    block_size = 50

    buffer_1 = np.asarray(img_origin)
    buffer_2 = np.asarray(img_enhance)

    lmn_1 = rgb2lmn(buffer_1)
    img_l1 = lmn_1[:,:,0]
    h1,w1 = img_l1.shape
    lmn_2 = rgb2lmn(buffer_2)
    img_l2 = lmn_2[:,:,0]
    h2,w2 = img_l2.shape


    eme1 = 0.
    block_amt = (h1//block_size) * (w1//block_size)
    for hx in range(0, h1-block_size, block_size):
        for wx in range(0, w1-block_size, block_size):
            block = img_l1[hx:(hx+block_size),wx:(wx+block_size)]
            eme1 += 20 * np.log((np.max(block)+0.0001)/(np.min(block)+0.0001)) / block_amt

    eme2 = 0.
    block_amt = (h1//block_size) * (w1//block_size)
    for hx in range(0, h2-block_size, block_size):
        for wx in range(0, w2-block_size, block_size):
            block = img_l2[hx:(hx+block_size),wx:(wx+block_size)]
            eme2 += 20 * np.log((np.max(block)+0.0001)/(np.min(block)+0.0001)) / block_amt

    
    eme = eme2 - eme1

    return eme, eme2, eme1




def compute_psnr(img1, img2):
    buffer_1 = np.asarray(img1)
    buffer_2 = np.asarray(img2)

    img1 = buffer_1.astype(np.float64) / 255.
    img2 = buffer_2.astype(np.float64) / 255.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 0
    return 10 * math.log10(1. / mse)

def cal_power_loss(img_dim,img_ori):
    buffer_dim = np.asarray(img_dim)
    avg_dim = np.average(buffer_dim)

    buffer_ori = np.asarray(img_ori)
    avg_ori = np.average(buffer_ori)

    inten_sup_ratio = (avg_ori - avg_dim) / avg_ori *100

    return inten_sup_ratio


if __name__ == '__main__':


    input_img_path = './other_method_result/testA_ori_/'
    input_img_names = []
    filenames = os.listdir(input_img_path)
    for filename in filenames:
        if(filename[-10:-4:1] == 'fake_C'):      
            input_img_names.append(filename[:-10:1])

    image_list = []
    f = open(input_img_path+"IQA.txt", "w")
    f.write(',Chondro,vsi,nde,eme,ssim,psnr,psr,eme_enhance,eme_ori,,Chang,vsi,nde,eme,ssim,psnr,psr,eme_enhance,eme_ori\n')
    print("=======================================================")
    item = 0.
    ssim_summa = 0.
    psnr_summa = 0.
    vsi_summa = 0.
    nde_summa = 0.
    eme_summa = 0.
    ps_rate_summa = 0.
    for filename in input_img_names:
        item += 1.
        print(str(item))
        os_filename = os.path.join(input_img_path, filename)
        os_filename_C = os_filename + 'fake_C.png'
        os_filename_P = os_filename + 'fake_P.png'
        os_filename_A = os_filename + 'real_A.png'
        
        print(os_filename_C)
        print(os_filename_P)
        print(os_filename_A)
        
        image1 = Image.open(os_filename_P)
        image2 = Image.open(os_filename_A)
        vsi = compute_vsi(image1,image2)
        vsi_summa += vsi
        nde = compute_nde(image1,image2)
        nde_summa += nde
        eme, eme_enhance, eme_ori = compute_eme(image1,image2)
        eme_summa += eme
        ssim_value = compare_ssim(image1, image2)
        ssim_summa += ssim_value
        psnr_value = compute_psnr(image1, image2)
        psnr_summa += psnr_value
        ps_rate_value = cal_power_loss(image1, image2)
        ps_rate_summa += ps_rate_value

        print('-------------Chondro-------------------')
        print('vsi:\t\t'+str(vsi)+'\tavg_vsi:\t'+str(vsi_summa/item))
        print('nde:\t\t'+str(nde)+'\tavg_nde:\t'+str(nde_summa/item))
        print('eme:\t\t'+str(eme)+'\tavg_eme:\t'+str(eme_summa/item)+"\teme_en:"+str(eme_enhance)+"\teme_ori:"+str(eme_ori))
        print('ssim:\t\t'+str(ssim_value)+'\tavg_ssim:\t'+str(ssim_summa/item))
        print('psnr:\t\t'+str(psnr_value)+'\tavg_psnr:\t'+str(psnr_summa/item))
        print('psr:\t\t'+str(ps_rate_value)+'\tavg_psr:\t'+str(ps_rate_summa/item))
        # f.write('\n'+str(filename)+'\n')
        # f.write("---------------------------------------------------\n")
        # f.write('Chondro\n')
        # f.write('vsi,nde,eme,ssim,psnr,psr,,eme_enhance,eme_ori\n')
        # f.write(str(vsi)+','+str(nde)+','+str(eme)+','+str(ssim_value)+','+str(psnr_value)+','+str(ps_rate_value)+',,'+str(eme_enhance)+','+str(eme_ori)+'\n')
        # f.write('avg\n')
        # f.write(str(vsi_summa/item)+','+str(nde_summa/item)+','+str(eme_summa/item)+','+str(ssim_summa/item)+','+str(psnr_summa/item)+','+str(ps_rate_summa/item)+'\n')
        f.write(str(filename)+',,'+str(vsi)+','+str(nde)+','+str(eme)+','+str(ssim_value)+','+str(psnr_value)+','+str(ps_rate_value)+','+str(eme_enhance)+','+str(eme_ori)+',,')

        image1 = Image.open(os_filename_C)
        image2 = Image.open(os_filename_A)
        vsi = compute_vsi(image1,image2)
        vsi_summa += vsi
        nde = compute_nde(image1,image2)
        nde_summa += nde
        eme, eme_enhance, eme_ori = compute_eme(image1,image2)
        eme_summa += eme
        ssim_value = compare_ssim(image1, image2)
        ssim_summa += ssim_value
        psnr_value = compute_psnr(image1, image2)
        psnr_summa += psnr_value
        ps_rate_value = cal_power_loss(image1, image2)
        ps_rate_summa += ps_rate_value

        print('--------------Chang--------------------')
        print('vsi:\t\t'+str(vsi)+'\tavg_vsi:\t'+str(vsi_summa/item))
        print('nde:\t\t'+str(nde)+'\tavg_nde:\t'+str(nde_summa/item))
        print('eme:\t\t'+str(eme)+'\tavg_eme:\t'+str(eme_summa/item)+"\teme_en:"+str(eme_enhance)+"\teme_ori:"+str(eme_ori))
        print('ssim:\t\t'+str(ssim_value)+'\tavg_ssim:\t'+str(ssim_summa/item))
        print('psnr:\t\t'+str(psnr_value)+'\tavg_psnr:\t'+str(psnr_summa/item))
        print('psr:\t\t'+str(ps_rate_value)+'\tavg_psr:\t'+str(ps_rate_summa/item))

        f.write(','+str(vsi)+','+str(nde)+','+str(eme)+','+str(ssim_value)+','+str(psnr_value)+','+str(ps_rate_value)+','+str(eme_enhance)+','+str(eme_ori)+'\n')
        # f.write("---------------------------------------------------\n")
        # f.write('Chang\n')
        # f.write('vsi,nde,eme,ssim,psnr,psr,,eme_enhance,eme_ori\n')
        # f.write(str(vsi)+','+str(nde)+','+str(eme)+','+str(ssim_value)+','+str(psnr_value)+','+str(ps_rate_value)+',,'+str(eme_enhance)+','+str(eme_ori)+'\n')
        # f.write('avg\n')
        # f.write(str(vsi_summa/item)+','+str(nde_summa/item)+','+str(eme_summa/item)+','+str(ssim_summa/item)+','+str(psnr_summa/item)+','+str(ps_rate_summa/item)+'\n')

    f.close()