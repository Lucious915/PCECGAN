# conda activate slic
# python3 val.py

import math
import glob
from PIL import Image
from SSIM_PIL import compare_ssim
import numpy as np
import numpy.matlib
import os
#niqe
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
import scipy
import scipy.ndimage
import scipy.special
import math


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
    XYZ[:,:,0] = XYZ[:,:,0] / 95.047            # ref_X =  95.047   Observer= 2, Illuminant= D65
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

def rgb2yuv(inputRGB):
    RGB = inputRGB/ 255.
    YUV = RGB
    YUV[:,:,0] = 0.299 * RGB[:,:,0] + 0.587 * RGB[:,:,1] + 0.114 * RGB[:,:,2]
    YUV[:,:,1] = -0.169 * RGB[:,:,0] - 0.331 * RGB[:,:,1] +0.5 * RGB[:,:,2]
    YUV[:,:,2] = 0.5 * RGB[:,:,0] - 0.419  * RGB[:,:,1] - 0.081 * RGB[:,:,2]


    return YUV

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)

def aggd_features(imdata):
    #flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata*imdata
    left_data = imdata2[imdata<0]
    right_data = imdata2[imdata>=0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
      gamma_hat = left_mean_sqrt/right_mean_sqrt
    else:
      gamma_hat = np.inf
    #solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
      r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
    else:
      r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1)*(gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    #solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm)**2);
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0/alpha)
    gam2 = scipy.special.gamma(2.0/alpha)
    gam3 = scipy.special.gamma(3.0/alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    #mean parameter
    N = (br - bl)*(gam2 / gam1)#*aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

def ggd_features(imdata):
    nr_gam = 1/prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq/E**2
    pos = np.argmin(np.abs(nr_gam - rho));
    return gamma_range[pos], sigma_sq

def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights

def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
      avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    return (image - mu_image)/(var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl+br)/2.0,
            alpha1, N1, bl1, br1,  # (V)
            alpha2, N2, bl2, br2,  # (H)
            alpha3, N3, bl3, bl3,  # (D1)
            alpha4, N4, bl4, bl4,  # (D2)
    ])

def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)

def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)

def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = np.int(patch_size)
    patches = []
    for j in range(0, h-patch_size+1, patch_size):
        for i in range(0, w-patch_size+1, patch_size):
            patch = img[j:j+patch_size, i:i+patch_size]
            patches.append(patch)

    patches = np.array(patches)
    
    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features

def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0: 
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]


    img = img.astype(np.float32)
    img_uint8 = Image.fromarray(img.astype(np.uint8))

    img2 = np.asarray(img_uint8.resize((int(h*0.5),int(w*0.5))))
    #scipy.misc.imresize(img, 0.5, interp='bicubic', mode='F')

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)


    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size/2)

    feats = np.hstack((feats_lvl1, feats_lvl2))# feats_lvl3))

    return feats

def niqe(img):


    img_Y = rgb2yuv(np.asarray(img))[:,:,0]


    patch_size = 72#96
    module_path = dirname(__file__)

    # TODO: memoize
    params = scipy.io.loadmat(join(module_path,'niqe', 'data', 'niqe_image_params.mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]


    M, N = img_Y.shape

    # assert C == 1, "niqe called with videos containing %d channels. Please supply only the luminance channel" % (C,)
    assert M > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"


    feats = get_patches_test_features(img_Y, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = ((pop_cov+sample_cov)/2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score





def save_IMG(img, os_F, name):
    img_nor = img/np.max(img)*255
    pil_img = Image.fromarray(img_nor.astype(np.uint8))
    pil_img.save(os_F+name)

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
    
    yuv = rgb2yuv(img)
    img_l = yuv[:,:,0]
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

def compute_vsi(img1, img2, os_F):
    C1 = 1
    C2 = 1
    C3 = 1
    Alpha = 0.40
    Beta = 0.02
    buffer_01 = np.asarray(img1)
    buffer_02 = np.asarray(img2)

    buffer_1 = buffer_01[5:-5,5:-5,:]
    buffer_2 = buffer_02[5:-5,5:-5,:]
    

    VSMap1 = sdsp(buffer_1)
    VSMap2 = sdsp(buffer_2)
    VSMap1 /= np.max(VSMap1)
    VSMap2 /= np.max(VSMap2)
    VSm = np.maximum(VSMap1,VSMap2)
    Svs = (2*np.multiply(VSMap1,VSMap2)+C1) / (VSMap1**2+VSMap2**2+C1)


    GMap1 = scharr_gradient_filter(buffer_1)
    GMap2 = scharr_gradient_filter(buffer_2)
    Sg = (2*np.multiply(GMap1,GMap2)+C2) / (GMap1**2+GMap2**2+C2)

    # save_IMG(GMap1,os_F,"fake_P_GM1.png")
    # save_IMG(GMap2,os_F,"fake_P_GM2.png")


    LMN1 = rgb2lmn(buffer_1)
    LMN2 = rgb2lmn(buffer_2)
    SM = (2*np.multiply(LMN1[:,:,1],LMN2[:,:,1])+C3) / (LMN1[:,:,1]**2+LMN2[:,:,1]**2+C3)
    SN = (2*np.multiply(LMN1[:,:,2],LMN2[:,:,2])+C3) / (LMN1[:,:,2]**2+LMN2[:,:,2]**2+C3)
    Sc = np.multiply(SM,SN)

    Sgc = np.multiply(Sg**Alpha,Sc**Beta)
    S = np.multiply(Svs,Sgc)

    # save_IMG(S,os_F,"fake_P_S.png")
    

    VSI = np.sum( np.multiply(S,VSm) ) / np.sum(VSm)

    return VSI

def compute_nde(img_enhance, img_origin):

    buffer_01 = np.asarray(img_origin)
    buffer_02 = np.asarray(img_enhance)
    buffer_1 = buffer_01[5:-5,5:-5,:]
    buffer_2 = buffer_02[5:-5,5:-5,:]

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
    block_size = 3

    buffer_01 = np.asarray(img_origin)
    buffer_02 = np.asarray(img_enhance)
    buffer_1 = buffer_01[5:-5,5:-5,:]
    buffer_2 = buffer_02[5:-5,5:-5,:]

    lmn_1 = rgb2yuv(buffer_1)
    img_l1 = lmn_1[:,:,0]
    h1,w1 = img_l1.shape
    lmn_2 = rgb2yuv(buffer_2)
    img_l2 = lmn_2[:,:,0]
    h2,w2 = img_l2.shape

    count = 0
    eme1 = 0.
    block_amt = (h1//block_size) * (w1//block_size)
    

    for hx in range(0, h1-block_size, block_size):
        for wx in range(0, w1-block_size, block_size):
            block = img_l1[hx:(hx+block_size),wx:(wx+block_size)]
            # print("--"+str(20*np.log((np.max(block)+0.0001)/(np.min(block)+0.0001))))
            eme1 += 20 * np.log((np.max(block)+0.0001)/(np.min(block)+0.0001)) / block_amt


    eme2 = 0.
    block_amt = (h2//block_size) * (w2//block_size)
    for hx in range(0, h2-block_size, block_size):
        for wx in range(0, w2-block_size, block_size):
            block = img_l2[hx:(hx+block_size),wx:(wx+block_size)]

            eme2 += 20 * np.log((np.max(block)+0.0001)/(np.min(block)+0.0001)) / block_amt
       
    # print("eme2:"+str(eme2)+"\teme1:"+str(eme1))
    eme = eme2 - eme1

    return eme,eme2,eme1




def compute_psnr(img1, img2):

    buffer_01 = np.asarray(img1)
    buffer_02 = np.asarray(img2)
    buffer_1 = buffer_01[5:-5,5:-5,:]
    buffer_2 = buffer_02[5:-5,5:-5,:]

    img1 = buffer_1.astype(np.float64) / 255.
    img2 = buffer_2.astype(np.float64) / 255.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 0
    return 10 * math.log10(1. / mse)

def cal_power_loss(img_dim,img_ori):
    buffer_dim = np.asarray(img_dim)
    # buffer_dim_r = buffer_dim[:,:,0]
    # buffer_dim_g = buffer_dim[:,:,1]
    # buffer_dim_b = buffer_dim[:,:,2]
    # buffer_dim_power_r = (0.0009/2073600)*(buffer_dim_r**2) + (0.035/2073600)*buffer_dim_r + (14.165/2073600)
    # buffer_dim_power_g = (0.0011/2073600)*(buffer_dim_g**2) + (0.0273/2073600)*buffer_dim_g + (14.541/2073600)
    # buffer_dim_power_b = (0.0022/2073600)*(buffer_dim_b**2) + (-0.0161/2073600)*buffer_dim_b + (15.241/2073600)
    # buffer_dim_power = buffer_dim_power_r + buffer_dim_power_g + buffer_dim_power_b
    # avg_dim = np.average(buffer_dim_power)
    avg_dim = np.average(buffer_dim)

    buffer_ori = np.asarray(img_ori)
    # buffer_ori_r = buffer_ori[:,:,0]
    # buffer_ori_g = buffer_ori[:,:,1]
    # buffer_ori_b = buffer_ori[:,:,2]
    # buffer_ori_power_r = (0.0009/2073600)*(buffer_ori_r**2) + (0.035/2073600)*buffer_ori_r + (14.165/2073600)
    # buffer_ori_power_g = (0.0011/2073600)*(buffer_ori_g**2) + (0.0273/2073600)*buffer_ori_g + (14.541/2073600)
    # buffer_ori_power_b = (0.0022/2073600)*(buffer_ori_b**2) + (-0.0161/2073600)*buffer_ori_b + (15.241/2073600)
    # buffer_ori_power = buffer_ori_power_r + buffer_ori_power_g + buffer_ori_power_b
    # avg_ori = np.average(buffer_ori_power)
    avg_ori = np.average(buffer_ori)

    inten_sup_ratio = (avg_ori - avg_dim) / avg_ori *100

    return inten_sup_ratio


if __name__ == '__main__':



    # input_img_path = './ablation/enlightening/test_25/images_nor/'

    input_img_path = './other_method_result/proposed_with_enhancer/images_tid__'
    input_img_names = []
    filenames = os.listdir(input_img_path)
    for filename in filenames:
        if(filename[-10:-4:1] == 'fake_B'):      
            input_img_names.append(filename[:-10:1])

    image_list = []
    f = open(input_img_path+"IQA_test_with_enhancer_P.csv", "w")
    f.write(',Chou,psnr,ssim,vsi,nde,niqe,psr,eme_enhance,eme_difference,eme_ori\n')
    print("=======================================================")
    item = 0.
    ssim_summa = 0.
    psnr_summa = 0.
    vsi_summa = 0.
    nde_summa = 0.
    eme_enh_summa = 0.
    eme_ori_summa = 0.
    ps_rate_summa = 0.
    niqe_summa = 0.


    for filename in input_img_names:
        item += 1.
        print(str(item))
        os_filename = os.path.join(input_img_path, filename)
        os_filename_B = os_filename + 'fake_B.png'
        os_filename_A = os_filename + 'real_A.png'
        
        print(os_filename_B)
        print(os_filename_A)
        
        image1 = Image.open(os_filename_B)
        image2 = Image.open(os_filename_A)

        vsi = compute_vsi(image1,image2,os_filename)
        vsi_summa += vsi
        nde = compute_nde(image1,image2)
        nde_summa += nde
        eme, eme_enhance, eme_ori = compute_eme(image1,image2)
        eme_enh_summa += eme_enhance
        eme_ori_summa += eme_ori
        ssim_value = compare_ssim(image1, image2)
        ssim_summa += ssim_value
        psnr_value = compute_psnr(image1, image2)
        psnr_summa += psnr_value
        ps_rate_value = cal_power_loss(image1, image2)
        ps_rate_summa += ps_rate_value

        niqe_value = niqe(image1)
        niqe_summa += niqe_value


        print('-------------Chou-------------------')
        print('vsi:\t\t'+str(vsi)+'\tavg_vsi:\t'+str(vsi_summa/item))
        print('nde:\t\t'+str(nde)+'\tavg_nde:\t'+str(nde_summa/item))
        print('eme:\t\t'+str(eme_enhance)+'\tavg_eme:\t'+str(eme_enh_summa/item)+"\teme_en:"+str(eme_enhance)+"\teme_ori:"+str(eme_ori))
        print('ssim:\t\t'+str(ssim_value)+'\tavg_ssim:\t'+str(ssim_summa/item))
        print('psnr:\t\t'+str(psnr_value)+'\tavg_psnr:\t'+str(psnr_summa/item))
        print('psr:\t\t'+str(ps_rate_value)+'\tavg_psr:\t'+str(ps_rate_summa/item))
        print('niqe:\t\t'+str(niqe_value)+'\tavg_niqe:\t'+str(niqe_summa/item))
        f.write(str(filename)+',,'+str(psnr_value)+','+str(ssim_value)+','+str(vsi)+','+str(nde)+','+str(niqe_value)+','+str(ps_rate_value)+','+str(eme_enhance)+','+str(eme)+','+str(eme_ori)+'\n')

