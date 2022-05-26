import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model import Discriminator, Generator, initialize_weights
from tqdm import tqdm
from load import customData
import numpy as np
import random
import os
from scipy.fftpack import fft, fftshift, ifft
def get_CC(next,real):
    cc_col = []
    for i in range(0, next.shape[0]):
        cov = np.cov(next[i, :], real[i, :], ddof=1)
        avgnext = np.cov(next[i, :])
        avgreal = np.cov(real[i, :])
        cc = cov[0, 1] / np.sqrt(avgnext * avgreal)
        cc_col.append(cc)
    # print(np.mean(cc))
    return np.mean(cc_col),np.max(cc_col),np.min(cc_col)


def getPSD(next,real):
    dt = 1 / 512
    t = np.arange(0, 2, dt)
    RMS=[]
    for i in range(0, next.shape[0]):
        # pxx_n, freqs = mlab.psd(x=next[i,0:512], Fs=512)
        # pxx_l, freqs = mlab.psd(x=real[i,0:512], Fs=512)
        # real_pxx_n=10 * np.log10(pxx_n)
        # real_pxx_l=10 * np.log10(pxx_l)
        real_pxx_n=getftt(next[i,:])
        real_pxx_l=getftt(real[i,:])
        up=real_pxx_n-real_pxx_l
        botom=real_pxx_l
        EEG_sq_up = np.square(up)
        MSR_X = np.sqrt(np.sum(EEG_sq_up) / EEG_sq_up.shape[0] )
        EEG_sq_bottom = np.square(botom)
        MSR_Y = np.sqrt(np.sum(EEG_sq_bottom) / EEG_sq_bottom.shape[0] )
        RMS.append(MSR_X/MSR_Y)

    print("psd",np.argmax(RMS),np.argmin(RMS))
    return np.mean(RMS),np.max(RMS),np.min(RMS)
    # # fig, (ax0, ax1, ax2,ax3) = plt.subplots(4, 1)
    # # ax0.plot(t, next[3, :])
    # # ax2.plot(t, real[3, :])
    # # pxx_n, freqs = ax1.psd(next[3, :], 1024, 2 / dt)
    # # pxx_l, freqs=ax3.psd(real[3, :], 1024, 2 / dt)
    # real_pxx_n=10 * np.log10(pxx_n)
    # real_pxx_l=10 * np.log10(pxx_l)
    # plt.show()
    # return  real_pxx


def get_RRMSE_t(next,real):
    MSR = []
    up = next - real
    # up = up * std_true
    for i in range(0,up.shape[0]):
        EEG_sq = np.square(real[i,:])
        MSR_X = np.sqrt(np.sum(EEG_sq) / 1024 )
        up_sq = np.square(up[i,:])
        MSR_up = np.sqrt(np.sum(up_sq) / 1024 )
        MSR.append(MSR_up / MSR_X)
    # print(np.mean(MSR))
    # print(MSR)
    print("t--",np.argmax(MSR),np.argmin(MSR))

    return  np.mean(MSR),np.max(MSR),np.min(MSR)


# def get_PSD_data(t):
#     num_fft = 1024
#     # generate original signal
#     Y = fft(t, num_fft)
#     Y = np.abs(Y)
#     # power spectrum
#     ps = Y ** 2 / num_fft
#     # power spectrum using correlate
#     cor_x = np.correlate(t, t, 'same')
#     cor_X = fft(cor_x, num_fft)
#     ps_cor = np.abs(cor_X)
#     ps_cor = ps_cor / np.max(ps_cor)
#     ps_cor=ps_cor+0.001
#     return  10*np.log10(ps_cor[:num_fft//2])


def getftt(t):
    y =t
    Y = fft(y)
    shift_Y = fftshift(Y)
    pos_Y_from_shift = shift_Y[shift_Y.size // 2:]
    return  np.abs(pos_Y_from_shift)**2/1024

def test(model,test_loader):
    real=[]
    next=torch.tensor([])
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            x, y = data
            x = x.view(-1,1,1024).cuda().float()
            y_hat = model(x)
            if len(real)==0:
                real=y
            else:
                real=torch.cat((real,y),dim=0)
            if len(next)==0:
                next=y_hat
            else:
                next=torch.cat((next,y_hat),dim=0)
    print(next.shape)
    return  next,real

def mix(SNR,EEG,EMG):
    data_mix=np.random.randn(EEG.shape[0],EEG.shape[1])
    for i in range(0,EEG.shape[0]):
        EEG_sq=np.square(EEG[i,:])
        MSR_X=np.sqrt(np.sum(EEG_sq)/1024)
        EMG_sq=np.square(EMG[i,:])
        MSR_N=np.sqrt(np.sum(EMG_sq)/1024)
        m=(MSR_X/MSR_N)*0.1**(0.1*SNR)
        data_mix[i,:]=EMG[i, :] * m+EEG[i,:]
    return data_mix

def shuffle(feats,labels):
    index = [i for i in range(labels.shape[0])]
    random.shuffle(index)
    labels = labels[index]
    feats=feats[index]
    return feats,labels

if __name__ == '__main__':
    # EEG=np.load('./data/EEG_3400_1024.npy')
    # EOG=np.load('./data/EOG_3400_1024.npy')
    # EOG,EEG=shuffle(EOG,EEG)
    # mix_EOG_2720=np.array([])
    # EEG_col_2720=np.array([])
    # mix_EOG_680=np.array([])
    # EEG_col_680=np.array([])
    # for SNR in range (0,10):
    #     SNR=SNR-7
    #     data_mix=mix(SNR=SNR,EEG=EEG,EMG=EOG)
    #     if mix_EOG_2720.size==0:
    #        mix_EOG_2720=data_mix[:2720,:]
    #     else:
    #        mix_EOG_2720=np.vstack((mix_EOG_2720,data_mix[:2720,:]))
    #
    #     if EEG_col_2720.size==0:
    #         EEG_col_2720=EEG[:2720,:]
    #     else:
    #         EEG_col_2720=np.vstack((EEG_col_2720,EEG[:2720,:]))
    #
    #     if mix_EOG_680.size==0:
    #         mix_EOG_680=data_mix[2720:,:]
    #     else:
    #         mix_EOG_680=np.vstack((mix_EOG_680,data_mix[2720:,:]))
    #
    #     if EEG_col_680.size==0:
    #         EEG_col_680=EEG[2720:,:]
    #     else:
    #         EEG_col_680=np.vstack((EEG_col_680,EEG[2720:,:]))
    # print(mix_EOG_2720.shape)
    # print(EEG_col_2720.shape)
    # print(EEG_col_680.shape)
    # print(mix_EOG_680.shape)
    # np.save("./data/mix_EOG_27200.npy",mix_EOG_2720)
    # np.save("./data/true_27200.npy",EEG_col_2720)
    # np.save("./data/mix_EOG_test_6800.npy",mix_EOG_680)
    # np.save("./data/true_test_6800.npy",EEG_col_680)


    mix_EEG_col_test=np.load('./data/mix_EOG_test_6800.npy')
    real_EEG_col_test=np.load('./data/true_test_6800.npy')


    test_mix_EEG=mix_EEG_col_test
    std_test=np.std(test_mix_EEG, ddof = 1)
    # test_mix_EEG=test_mix_EEG/std_test
    # print(std_test)

    test_true_EEG=real_EEG_col_test
    std_true_test=np.std(test_true_EEG, ddof = 1)
    # test_true_EEG=test_true_EEG/std_true_test
    RRMSE_t,RRMSE_t_max,RRMSE_t_min=get_RRMSE_t(test_mix_EEG, test_true_EEG)
    print("RRMSE_t avg:{} max:{} min:{}".format(RRMSE_t,RRMSE_t_max,RRMSE_t_min))
    CC,CC_max,CC_min=get_CC(test_mix_EEG, test_true_EEG)
    print("CC avg:{} max:{} min:{}".format(CC,CC_max,CC_min))
    RRMSE_f,RRMSE_f_max,RRMSE_f_min=getPSD(test_mix_EEG, test_true_EEG)
    print("psd RRMSE_f avg:{} max:{} min:{}".format(RRMSE_f,RRMSE_f_max,RRMSE_f_min))


