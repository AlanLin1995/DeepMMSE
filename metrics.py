''' metric computation including PSNR and SSIM '''
import torch
import numpy as np
from scipy import signal


def psnr(img1,img2):
    PIXEL_MAX = 1
    img1 = torch.clamp(img1,min = 0, max = PIXEL_MAX)
    img2 = torch.clamp(img2,min = 0, max = PIXEL_MAX)
    mse = torch.mean((img1.cpu() - img2.cpu()) ** 2).numpy()
    return 10 * np.log10(PIXEL_MAX **2 / mse)

def aver_psnr(img1,img2):
    ''' For images with same size and stored by a matrix'''
    PSNR = 0
    assert img1.size() == img2.size()
    for i in range(img1.size()[0]):	
        PSNR += psnr(img1[i,...], img2[i,...])
    return PSNR / img1.size()[0]

def aver_psnr_ds(img1, img2):
    ''' For images with different size and stored by a list'''
    PSNR = 0
    for i in range(len(img1)):
        PSNR += psnr(img1[i], img2[i])
    return PSNR / len(img1)


def aver_ssim(img1,img2):
    SSIM = 0
    assert img1.size() == img2.size()
    for i in range(img1.size()[0]):
        SSIM += ssim(img1[i,...], img2[i,...])
    return SSIM / img1.size()[0]

def aver_ssim_ds(img1, img2):
    ''' For images with different size and stored by a list'''
    SSIM = 0
    for i in range(len(img1)):
        SSIM += ssim(img1[i], img2[i])
    return SSIM / len(img1)

def ssim(img1, img2, cs_map=False):
    if isinstance(img1, torch.Tensor):
        img1 = img1.squeeze()
        img2 = img2.squeeze()
        img1 = img1.cpu().numpy()
        img2 = img2.cpu().numpy()
        if np.max(img2) < 2:
            img1 = img1 * 255
            img2 = img2 * 255

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
        return ssim.mean()

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

