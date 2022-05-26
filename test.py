from tensorboardX import SummaryWriter
from model import Discriminator, Generator
import torch
import numpy as np
from torchsummary import summary
import scipy.io as sio
import scipy.io as sio
import numpy as np
import os
import random
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model import Discriminator, Generator, initialize_weights
from tqdm import tqdm
from load import customData
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from scipy.fftpack import fft, fftshift, ifft
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import numpy as np
import matplotlib.pyplot as plt



class customData(Dataset):
    def __init__(self,feat,label):
        self.feats=feat
        self.labels=label
        # self.feats=np.load(feat_path)
        # self.feats=self.feats.reshape(self.feats.shape[0],32,60,128)
        # self.labels=np.load(label_path)
        # self.labels=culate(self.labels)
        # print(self.feats.shape)
        # print(self.labels.shape)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        item=self.feats[index]
        label=self.labels[index]
        return item,label
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
LEARNING_RATE_dis = 6e-5  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 1120


# NOISE_DIM = 100
NUM_EPOCHS = 50
def test(model,test_loader):
    real=torch.tensor([])
    next=torch.tensor([])
    model.eval()
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


def get_PSD_data(t):
    num_fft = 1024
    # generate original signal
    Y = fft(t, num_fft)
    Y = np.abs(Y)
    # power spectrum
    ps = Y ** 2 / num_fft
    # power spectrum using correlate
    cor_x = np.correlate(t, t, 'same')
    cor_X = fft(cor_x, num_fft)
    ps_cor = np.abs(cor_X)
    ps_cor = ps_cor / np.max(ps_cor)
    ps_cor=ps_cor+0.001
    return  10*np.log10(ps_cor[:num_fft//2])


def getftt(t):
    y =t
    Y = fft(y)
    shift_Y = fftshift(Y)
    pos_Y_from_shift = shift_Y[shift_Y.size // 2:]
    return  np.abs(pos_Y_from_shift)**2/1024


if __name__ == '__main__':
    mix_EEG=np.load('./data/11200_mix_EEG_test.npy')
    EEG=np.load('./data/11200_real_EEG_test.npy')
    test_mix_EEG=mix_EEG
    test_true_EEG=EEG
    std_mix=np.std(test_mix_EEG, ddof = 1)
    test_mix_EEG=test_mix_EEG/std_mix
    std_true=np.std(test_true_EEG, ddof = 1)
    test_true_EEG=test_true_EEG/std_true

    """
       ############################################################
       模型，优化器定义
       ############################################################
       """
    gen = Generator(channels_noise=1).to(device)
    disc = Discriminator().to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_dis, betas=(0.5, 0.999))
    """
    ############################################################
    判预加载模型
    ############################################################
    """
    save_root='./train'
    dir_gen=os.path.join(save_root,'gen')
    dir_disc=os.path.join(save_root,'disc')
    epochpoint=0
    if os.path.exists('./train/gen/'):
        all = os.listdir(dir_gen)
        if len(all)>0:
            checkpoint=torch.load(os.path.join(dir_gen,all[len(all)-1]))
            gen.load_state_dict(checkpoint['model'])
            opt_gen.load_state_dict(checkpoint['optimizer'])
            epochpoint = checkpoint['epoch']
            print("gen模型加载{}".format(all[len(all)-1]))

    if os.path.exists(dir_disc):
        all = os.listdir(dir_disc)
        if len(all)>0:
            checkpoint=torch.load(os.path.join(dir_disc,all[len(all)-1]))
            disc.load_state_dict(checkpoint['model'])
            opt_disc.load_state_dict(checkpoint['optimizer'])
            epochpoint = checkpoint['epoch']
            print("disc模型加载{}".format(all[len(all)-1]))
    avg_rmst=[]
    avg_rmsf=[]
    avg_cc=[]
    for i in range(0,10):

        # SNR=-6
        # mix_EEG=mix(SNR=SNR,EEG=EEG,EMG=EMG)
        # mix_EEG,EEG=shuffle(mix_EEG,EEG)

        # train_mix_EEG=mix_EEG[0:4478,:]
        # std_train=np.std(train_mix_EEG, ddof = 1)
        # train_mix_EEG=train_mix_EEG/std_train
        #
        # train_true_EEG=EEG[0:4478,:]
        # std_true=np.std(train_true_EEG, ddof = 1)
        # train_true_EEG=train_true_EEG/std_true


        test_mix_EEG_batch=test_mix_EEG[i*1120:(i+1)*1120,:]
        test_true_EEG_batch=test_true_EEG[i*1120:(i+1)*1120,:]
        print("ww",i*1120,(i+1)*1120)


        dataset=customData(test_mix_EEG_batch,test_true_EEG_batch)
        dataloader = DataLoader(dataset, batch_size=560,shuffle=False)
        next,real=test(gen,dataloader)

        next=next.cpu().numpy()*std_true
        real=real.cpu().numpy()*std_true
        print(next.shape)


        RRMSE_t,RRMSE_t_max,RRMSE_t_min=get_RRMSE_t(next,real)
        print("RRMSE_t avg:{} max:{} min:{}".format(RRMSE_t,RRMSE_t_max,RRMSE_t_min))
        CC,CC_max,CC_min=get_CC(next,real)
        print("CC avg:{} max:{} min:{}".format(CC,CC_max,CC_min))
        RRMSE_f,RRMSE_f_max,RRMSE_f_min=getPSD(next,real)
        print("psd RRMSE_f avg:{} max:{} min:{}".format(RRMSE_f,RRMSE_f_max,RRMSE_f_min))
        if len(avg_rmst)==0:
            avg_rmst=np.array([RRMSE_t,RRMSE_t_max,RRMSE_t_min])
        else:
            avg_rmst=np.vstack((avg_rmst,np.array([RRMSE_t,RRMSE_t_max,RRMSE_t_min])))

        if len(avg_rmsf)==0:
            avg_rmsf=np.array([RRMSE_f,RRMSE_f_max,RRMSE_f_min])
        else:
            avg_rmsf=np.vstack((avg_rmsf,np.array([RRMSE_f,RRMSE_f_max,RRMSE_f_min])))

        if len(avg_cc)==0:
            avg_cc=np.array([CC,CC_max,CC_min])
        else:
            avg_cc=np.vstack((avg_cc,np.array([CC,CC_max,CC_min])))
    print(avg_rmst.shape)


    print('RRMSE_t avg{},psd RRMSE_f{},CC{}'.format(np.mean(avg_rmst[:,0],axis=0),np.mean(avg_rmsf[:,0],axis=0),np.mean(avg_cc[:,0],axis=0)))
    t=np.array([-7,-6,-5,-4,-3,-2,-1,0,1,2])
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3,figsize=(15,4))
    fig.subplots_adjust(wspace=0.3)
    ax0.plot(t, avg_rmst[:,0],"y-.d",label="avg")
    ax0.plot(t,avg_rmst[:,1],"b-.1",label="max")
    ax0.plot(t,avg_rmst[:,2],"g-.2",label="min")

    ax0.set_title('RRMSE temporal')

    ax0.set_xlabel("SNR(dB)")

    ax1.plot(t,avg_rmsf[:,0],"y-.d",label='avg')
    ax1.plot(t,avg_rmsf[:,1],"b-.1",label='max')
    ax1.plot(t,avg_rmsf[:,2],"g-.2",label='min')

    ax1.set_title('RRMSE spectral')
    ax1.set_xlabel("SNR(dB)")
    ax2.plot(t, avg_cc[:,0],"y-.d",label='avg')
    ax2.plot(t,avg_cc[:,1],"b-.1",label='max')
    ax2.plot(t,avg_cc[:,2],"g-.2",label='min')

    ax2.set_title('CC')
    ax2.set_xlabel("SNR(dB)")
    plt.legend()
    plt.show()


    # rmst
    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(get_PSD_data(test_mix_EEG[116,:]),"b",label='NoiseEEG')
    # ax[0].plot(get_PSD_data(real[116,:]),"r",label='Raw')
    # ax[0].plot(get_PSD_data(next[116,:]),"g",label='Denoise')
    # ax[0].set_title('Worst Result')
    # ax[0].set_ylabel("PSD(dB)")
    # # ax[0].set_xlabel("Frequency(HZ)")
    #
    # ax[1].plot(get_PSD_data(test_mix_EEG[233,:]),"b",label='NoiseEEG')
    # ax[1].plot(get_PSD_data(real[233,:]),"r",label='Raw')
    # ax[1].plot(get_PSD_data(next[233,:]),"g",label='Denoise')
    # ax[1].set_title('Best Result')
    # ax[1].set_ylabel("PSD(dB)")
    # ax[1].set_xlabel("Frequency(HZ)")
    # plt.legend()
    # plt.show()





    # plt.plot(getftt(test_mix_EEG[343,:]),color="b",label='NoiseEEG')
    # plt.plot(getftt(real[343,:]),color="r",label='Raw')
    # plt.plot(getftt(next[343,:]),color="g",label='Denoise')


    #
    # fig,ax=plt.subplots(2,1)
    # ax[0].set_title('Worst Result')
    # ax[0].plot(test_mix_EEG[104,:],color="b",label='NoiseEEG')
    # ax[0].plot(real[104,:],color="r",label='Raw')
    # ax[0].plot(next[104,:],color="g",label='Denoise')
    # ax[0].set_ylabel("Amplitude(uV)")
    #
    # ax[1].plot(test_mix_EEG[488,:],"b",label='NoiseEEG')
    # ax[1].plot(real[488,:],"r",label='Raw')
    # ax[1].plot(next[488,:],"g",label='Denoise')
    # ax[1].set_title('Best Result')
    # ax[1].set_ylabel("Amplitude(uV)")
    # plt.legend()
    # plt.show()
    fig.savefig("avg.tiff",dpi=600)
