import numpy as np
import cv2
import torch
from torch.optim import Adam
from unet import network
import os
import scipy.io as sio
from datetime import datetime
import util
import metrics


def train(SAVE_PATH, file_list, epoch, sigma, num_mask, imsize, Epsilon, gamma=[0.05, 0.25], save_mat=False):
    n_file = len(file_list)

    MODEL_PATH = SAVE_PATH + 'model/'
    IMAGE_PATH = SAVE_PATH + 'image/'
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(IMAGE_PATH, exist_ok=True)

    # measurements
    # bipolar mask
    probability = torch.ones(num_mask, n_file, 1, imsize, imsize) * 0.5
    mask = torch.bernoulli(probability, out=None) * 2 - 1
    #   # illumination mask
    #    mask = np.exp(1j*2*np.pi*np.random.rand(num_mask, 1, 1, imsize, imsize))

    # capture data
    input_list = []
    label_list = []
    norm_list = []
    phi_list = []
    im_list = []
    sigma_w_list = []
    for i in file_list:
        input_, label, norm, phi, im, sigma_w = util.imread(i, SAVE_PATH, sigma, num_mask, imsize, mask)
        input_list.append(input_)
        label_list.append(label)
        norm_list.append(norm)
        phi_list.append(phi)
        im_list.append(im)
        sigma_w_list.append(sigma_w)
    feature = torch.cat(input_list, dim=-3)  # [bs, c, w, h]
    label = torch.cat(label_list, dim=-3)  # [r, bs, c, w, h]
    norm = torch.cat(norm_list, dim=-3)
    phi = torch.cat(phi_list, dim=-3)
    im = torch.cat(im_list, dim=-3)
    if len(sigma_w_list) > 1:
        sigma_w = torch.cat(sigma_w_list, dim=0)
    else:
        sigma_w = sigma_w_list[0]
    sigma_mean = torch.mean(sigma_w)
    print('sigma_w: ', sigma_mean)

    _, c, w, h = list(im.shape)
    c = c // n_file  # gray: 1   color: 3
    feature = feature.cuda()  # input
    norm = norm.cuda()  # ground truth

    learning_rate = 1e-4
    num_itera = 100000
    if sigma == 9 or sigma == 20:
        print_freq = 10000
        learning_rate = learning_rate * 3
    elif sigma == 27 or sigma == 15:
        print_freq = 10000
        learning_rate = learning_rate * 2
    elif sigma == 81 or sigma == 10:
        print_freq = 10000
        learning_rate = learning_rate * 1

    # capture model
    model = network(c, n_file, num_mask, sigma)
    if epoch > 0:
        model.load_state_dict(torch.load(MODEL_PATH + 'checkpoint_' + '%d' % epoch + '.pth')['model_state_dict'])
    model = model.to(0)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # loss function
    def dip_loss(output, target):
        output = util.A_torch(output, num_mask, imsize, mask)  # [r, bs, c, w, h]
        output = torch.sqrt(output[..., 0] ** 2 + output[..., 1] ** 2 + 1e-8)  # **2 if Gaussian noise

        all_one = torch.ones(num_mask, n_file, 1, imsize, imsize).cuda()
        # bernoulli mask
        #        probability = torch.ones(num_mask, n_file, 1, imsize, imsize)*0.9
        #        loss_mask = torch.bernoulli(probability, out=None).cuda()
        # normal mask
        #        loss_mask = torch.rand(num_mask, n_file, 1, imsize, imsize).cuda()
        # gaussian mask
        loss_mask = torch.normal(mean=0.5, std=torch.ones(num_mask, n_file, 1, imsize, imsize) * 0.1).cuda()

        # norm loss
        loss = torch.sum(loss_mask * (output - target) ** 2) / torch.sum(loss_mask) * torch.sum(all_one)

        return loss

    # train
    clears = []
    for n in range(n_file):
        clears.append([])
    i = epoch
    while True:
        model.train()
        model.zero_grad()
        clear = model(feature)  # [bs, c, w, h]
        residual = dip_loss(clear, norm)

        loss = residual
        retain = False

        optimizer.zero_grad()
        loss.backward(retain_graph=retain)
        optimizer.step()

        #        if (i + 1) % 10000 == 0:
        #            torch.save({
        #                'epoch': i + 1,
        #                'model_state_dict': model.state_dict(),
        #                'optimizer_state_dict': optimizer.state_dict(),
        #                'loss': residual,
        #            }, MODEL_PATH + 'checkpoint_' + str(i + 1) + '.pth')

        mean_num = 50
        if i == 0 or (i + 1) % print_freq == 0:
            now = datetime.now()
            sum_ = np.zeros([n_file, w, h, c], dtype=np.float32)
            with torch.no_grad():
                for j in range(mean_num):
                    clear = model(feature)
                    optimizer.zero_grad()
                    clear_cpu = clear.cpu().detach().numpy()
                    clear_cpu = np.transpose(clear_cpu, [0, 2, 3, 1])
                    clear_cpu = np.concatenate(np.split(clear_cpu, n_file, axis=3), axis=0)
                    sum_ += clear_cpu
                for k in range(n_file):
                    clear = sum_[k] / mean_num  # [w, h, c]
                    im_ = im[k, ...]  # [c, w, h]
                    clear_ = torch.Tensor(np.transpose(clear, [2, 0, 1])).cuda()  # [c, w, h]
                    psnr, ssim = metrics.aver_psnr(clear_, im_), metrics.aver_ssim(clear_, im_)
                    print("loss in ", i + 1, ": ", residual.item(), ' ', round(psnr, 2), '/', round(ssim, 3), ' ',
                          now.strftime("%H:%M:%S"), sep='')

                    cv2.imwrite(IMAGE_PATH + str(k) + '_' + str(i + 1) + '_' + str(round(psnr, 2)) + '_' +
                                str(round(ssim, 3)) + '.png', np.int32(clear * 255))
                    if save_mat:
                        sio.savemat(IMAGE_PATH + str(k) + '_' + str(i + 1) + '_' + str(round(psnr, 2)) + '_' +
                                    str(round(ssim, 3)) + '.mat', {"x": clear})

                loss_file = open(SAVE_PATH + 'loss.txt', 'a')
                loss_file.write("\n" + str(residual.item()))
                loss_file.close()

        norm_num = norm.numel()
        stop_value = norm_num * (sigma_mean + Epsilon) ** 2

        if residual < stop_value * 1.01 and len(clears[0]) == 0:
            now = datetime.now()
            sum_ = np.zeros([n_file, w, h, c], dtype=np.float32)
            with torch.no_grad():
                for j in range(mean_num):
                    clear = model(feature)
                    optimizer.zero_grad()
                    clear_cpu = clear.cpu().detach().numpy()
                    clear_cpu = np.transpose(clear_cpu, [0, 2, 3, 1])
                    clear_cpu = np.concatenate(np.split(clear_cpu, n_file, axis=3), axis=0)
                    sum_ += clear_cpu
                for k in range(n_file):
                    clear = sum_[k] / mean_num  # [w, h, c]
                    clears[k].append(clear)

                    im_ = im[k, ...]  # [c, w, h]
                    clear_ = torch.Tensor(np.transpose(clear, [2, 0, 1])).cuda()  # [c, w, h]
                    psnr, ssim = metrics.aver_psnr(clear_, im_), metrics.aver_ssim(clear_, im_)
                    print("+ loss in ", i + 1, ": ", residual.item(), ' ', round(psnr, 2), '/', round(ssim, 3), ' ',
                          now.strftime("%H:%M:%S"), sep='')

                    cv2.imwrite(IMAGE_PATH + str(k) + '_' + str(i + 1) + '_' + str(round(psnr, 2)) + '_' +
                                str(round(ssim, 3)) + '.png', np.int32(clear * 255))
                    if save_mat:
                        sio.savemat(IMAGE_PATH + str(k) + '_' + str(i + 1) + '_' + str(round(psnr, 2)) + '_' +
                                    str(round(ssim, 3)) + '.mat', {"x": clear})

        if residual < stop_value and len(clears[0]) == 1:
            now = datetime.now()
            sum_ = np.zeros([n_file, w, h, c], dtype=np.float32)
            with torch.no_grad():
                for j in range(mean_num):
                    clear = model(feature)
                    optimizer.zero_grad()
                    clear_cpu = clear.cpu().detach().numpy()
                    clear_cpu = np.transpose(clear_cpu, [0, 2, 3, 1])
                    clear_cpu = np.concatenate(np.split(clear_cpu, n_file, axis=3), axis=0)
                    sum_ += clear_cpu
                for k in range(n_file):
                    clear = sum_[k] / mean_num  # [w, h, c]
                    clears[k].append(clear)

                    im_ = im[k, ...]  # [c, w, h]
                    clear_ = torch.Tensor(np.transpose(clear, [2, 0, 1])).cuda()  # [c, w, h]
                    psnr, ssim = metrics.aver_psnr(clear_, im_), metrics.aver_ssim(clear_, im_)
                    print("0 loss in ", i + 1, ": ", residual.item(), ' ', round(psnr, 2), '/', round(ssim, 3), ' ',
                          now.strftime("%H:%M:%S"), sep='')

                    cv2.imwrite(IMAGE_PATH + str(k) + '_' + str(i + 1) + '_' + str(round(psnr, 2)) + '_' +
                                str(round(ssim, 3)) + '.png', np.int32(clear * 255))
                    if save_mat:
                        sio.savemat(IMAGE_PATH + str(k) + '_' + str(i + 1) + '_' + str(round(psnr, 2)) + '_' +
                                    str(round(ssim, 3)) + '.mat', {"x": clear})

        if residual < stop_value * 0.99 or (i + 1) == num_itera:
            now = datetime.now()
            sum_ = np.zeros([n_file, w, h, c], dtype=np.float32)
            with torch.no_grad():
                for j in range(mean_num):
                    clear = model(feature)
                    optimizer.zero_grad()
                    clear_cpu = clear.cpu().detach().numpy()
                    clear_cpu = np.transpose(clear_cpu, [0, 2, 3, 1])
                    clear_cpu = np.concatenate(np.split(clear_cpu, n_file, axis=3), axis=0)
                    sum_ += clear_cpu
                for k in range(n_file):
                    clear = sum_[k] / mean_num  # [w, h, c]
                    clears[k].append(clear)

                    im_ = im[k, ...]  # [c, w, h]
                    clear_ = torch.Tensor(np.transpose(clear, [2, 0, 1])).cuda()  # [c, w, h]
                    psnr, ssim = metrics.aver_psnr(clear_, im_), metrics.aver_ssim(clear_, im_)
                    print("- loss in ", i + 1, ": ", residual.item(), ' ', round(psnr, 2), '/', round(ssim, 3), ' ',
                          now.strftime("%H:%M:%S"), sep='')

                    cv2.imwrite(IMAGE_PATH + str(k) + '_' + str(i + 1) + '_' + str(round(psnr, 2)) + '_' +
                                str(round(ssim, 3)) + '.png', np.int32(clear * 255))
                    if save_mat:
                        sio.savemat(IMAGE_PATH + str(k) + '_' + str(i + 1) + '_' + str(round(psnr, 2)) + '_' +
                                    str(round(ssim, 3)) + '.mat', {"x": clear})

                    # finnal result
                    clears_sum = np.zeros([w, h, c], dtype=np.float32)
                    for z in range(len(clears[k])):
                        clears_sum += clears[k][z]
                    clear = clears_sum / len(clears[k])

                    clear_ = torch.Tensor(np.transpose(clear, [2, 0, 1])).cuda()  # [c, w, h]
                    psnr, ssim = metrics.aver_psnr(clear_, im_), metrics.aver_ssim(clear_, im_)
                    print("best!", round(psnr, 2), '/', round(ssim, 3), ' ',
                          now.strftime("%H:%M:%S"), sep='')

                    cv2.imwrite(IMAGE_PATH + str(k) + '_' + 'best' + '_' + str(round(psnr, 2)) + '_' +
                                str(round(ssim, 3)) + '.png', np.int32(clear * 255))
                    if save_mat:
                        sio.savemat(IMAGE_PATH + str(k) + '_' + 'best' + '_' + str(round(psnr, 2)) + '_' +
                                    str(round(ssim, 3)) + '.mat', {"x": clear})

            torch.save({
                'epoch': i + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': residual,
            }, MODEL_PATH + 'checkpoint_best.pth')
            break
        i += 1
    return psnr, ssim


if __name__ == '__main__':
    save_mat = False
    num_mask = 1
    imsize = 128
    sigma = 9  # Gaussian or Poisson
    epoch = 0
    Epsilon = 1e-3  # 1e-3 for natural(Pn9); 5e-4 for natural(Pn27) 1e-4 for others
    filePath = './natural/Image/'
    filenames = os.listdir(filePath)
    PSNR, SSIM = 0, 0
    for i in range(len(filenames)):
        file_list = [filePath + filenames[i]]
        SAVE_PATH = './natural/bipolar_' + str(num_mask) + '/Poisson' + str(sigma) + '/' + filenames[i][:-4] + '/'
        psnr, ssim = train(SAVE_PATH, file_list, epoch, sigma, num_mask, imsize, Epsilon, save_mat=save_mat)
        PSNR, SSIM = PSNR + psnr, SSIM + ssim
    print(PSNR / len(filenames), SSIM / len(filenames))
