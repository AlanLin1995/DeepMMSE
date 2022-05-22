import numpy as np
import random
# from PIL import Image
import os
import cv2
import math
import scipy.io as sio
# import matplotlib.pyplot as pl
import torch
import torch.nn.functional as F


# from scipy.misc import imresize


def get_filter(type=9):
    file = sio.loadmat('./testset/filter.mat')
    kernel = file['kernel' + str(type)]
    sigma = file['noise' + str(type)]
    size = file['size' + str(type)]
    return np.float32(kernel), np.squeeze(sigma), np.squeeze(np.floor_divide(size, 2))


def imread_infer(filename):
    arr = sio.loadmat(filename)['im_blurred']
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :, np.newaxis]
    else:
        arr = arr[np.newaxis, :, :, :]
    arr = np.transpose(arr, [0, 3, 1, 2])
    img = torch.Tensor(arr)
    return img


##################
# for uniform mask (CDP)
# def A_numpy(x, num_mask, imsize, mask):
#    m = mask
#    x = np.repeat(x[np.newaxis, ...], num_mask, axis=0)
#    Ax = np.fft.fft2(x*m)/imsize 
#    return Ax
#
# def A_torch(x, num_mask, imsize, mask):
#    mask_r, mask_i = np.real(mask).astype('float32'), np.imag(mask).astype('float32')
#    mask_r, mask_i = torch.tensor(mask_r).cuda(), torch.tensor(mask_i).cuda()
#    x = x.repeat(num_mask,1,1,1,1)
#    x = torch.stack((x*mask_r, x*mask_i), dim=-1)
#    Ax = torch.fft(x, signal_ndim=2, normalized=False)/imsize 
#    return Ax
################## 

##################
# for bipolar mask
def A_numpy(x, num_mask, imsize, mask):
    m = mask.numpy()
    x = np.repeat(x[np.newaxis, ...], num_mask, axis=0)
    Ax = np.fft.fft2(x * m) / imsize
    return Ax


def A_torch(x, num_mask, imsize, mask):
    m = mask.cuda()
    x = x.repeat(num_mask, 1, 1, 1, 1)
    Ax = torch.rfft(x * m, signal_ndim=2, normalized=False, onesided=False) / imsize
    return Ax


##################

def Poisson_noise(norm, sigma):
    alpha = sigma / 255.
    r, bs, c, w, h = norm.size()
    #    torch.manual_seed(1)
    noise = torch.randn(r, bs, c, w, h).cuda()
    intensity_noise = alpha * norm * noise
    y = norm ** 2 + intensity_noise
    y = y * (y > 0).float()
    y = torch.sqrt(y + 1e-8)
    err = y - norm
    sigma_w = torch.std(err)
    return sigma_w, y


# def Gaussian_noise(norm, sigma):
#    sigma = sigma/255.
#    r, bs, c, w, h = norm.size()    
##    torch.manual_seed(1)
#    noise = sigma*torch.randn(r, bs, c, w, h).cuda()
#    y = norm + noise
#    return y

def Gaussian_noise(norm, SNR):
    norm = norm ** 2
    r, bs, c, w, h = norm.shape
    noise = np.random.randn(r, bs, c, w, h)
    noise = noise - np.mean(noise)
    norm_power = np.linalg.norm(norm) ** 2 / norm.size
    noise_variance = norm_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    y = norm + noise
    sigma_w = np.std(noise)
    return sigma_w, y


def imread(filename, model_path, sigma, num_mask, imsize, mask):
    img_prex = filename[filename.rfind("/"):filename.rfind(".")]
    print(img_prex)

    # im = np.array(cv2.imread(filename, -1), dtype=np.float32)/255.

    #    im = sio.loadmat(os.path.join(filename))
    #    im = im['x'].astype('float32')

    # imresize
    # im = imresize(im, [imsize, imsize], 'bilinear', mode='F')
    im = cv2.imread(filename, -1)
    im = np.array(im, dtype=np.float32) / 255.

    if im.ndim == 2:
        im = im[np.newaxis, :, :, np.newaxis]
    else:
        im = im[np.newaxis, :, :, :]
    im = np.transpose(im, [0, 3, 1, 2])  # [bs, c, w, h]

    #    im_hat = np.fft.fft2(im)
    im_hat = A_numpy(im, num_mask, imsize, mask)  # [r, bs, c, w, h]
    a = np.real(im_hat)
    b = np.imag(im_hat)
    norm = np.abs(im_hat)
    phi = np.arctan2(b, a)
    #    np.random.seed(1) # fix input for retraining
    bs, c, w, h = im.shape
    randnoise = np.random.normal(scale=0.1, size=(bs, c, w, h))  # [bs, c, w, h]
    #    randnoise = np.random.randn(im.shape[0], im.shape[1], im.shape[2], im.shape[3])   # [bs, c, w, h]
    sio.savemat(model_path + img_prex + '_input.mat',
                {'noise': randnoise, 'phi': phi, 'norm': norm, 'im': im})
    label = torch.Tensor(norm).cuda()
    input_ = torch.Tensor(randnoise).cuda()
    phi = torch.Tensor(phi).cuda()

    # add Gaussian noise
    #    sigma_w, norm = Gaussian_noise(norm, sigma)
    #    sigma_w = torch.tensor(sigma_w).cuda()

    norm = torch.Tensor(norm).cuda()

    # add Poisson noise
    sigma_w, norm = Poisson_noise(norm, sigma)

    im = torch.Tensor(im).cuda()  # [bs, c, w, h]
    return input_, label, norm, phi, im, sigma_w



def load_np_image(path, is_scale=True):
    img = cv2.imread(path, -1)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    if is_scale:
        img = np.array(img).astype(np.float32) / 255.
    return img


