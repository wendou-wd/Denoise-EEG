"""
Training of DCGAN network on MNIST dataset with Discriminator
and Generator imported from models.py
"""
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

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
LEARNING_RATE_dis = 3e-5  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 64


# NOISE_DIM = 100
NUM_EPOCHS = 25

"""
############################################################
数据加载，训练集，测试集，验证集
############################################################
"""
# EEG=np.load('./data/EEG_5598_1024.npy')
# EMG=np.load('./data/EMG_1024.npy')
# mix_EEG_col=[]
# real_EEG_col=[]
# 
# mix_EEG_col_test=[]
# real_EEG_col_test=[]
# for SNR in range(0,10):
#     mix_EEG=mix(SNR=SNR-7,EEG=EEG,EMG=EMG)
#     mix_EEG,EEG=shuffle(mix_EEG,EEG)
#     if len(real_EEG_col)==0:
#         real_EEG_col=EEG[:4478,:]
#     else:
#         real_EEG_col=np.vstack((real_EEG_col,EEG[:4478,:]))
#     if len(mix_EEG_col)==0:
#         mix_EEG_col=mix_EEG[:4478]
#     else:
#         mix_EEG_col=np.vstack((mix_EEG_col,mix_EEG[:4478,:]))
# 
#     if len(mix_EEG_col_test)==0:
#         mix_EEG_col_test=mix_EEG[4478:,:]
#     else:
#         mix_EEG_col_test=np.vstack((mix_EEG_col_test,mix_EEG[4478:,:]))
#     if len(real_EEG_col_test)==0:
#         real_EEG_col_test=EEG[4478:,:]
#     else:
#         real_EEG_col_test=np.vstack((real_EEG_col_test,EEG[4478:,:]))
# print(mix_EEG_col.shape)
# print(real_EEG_col.shape)
# print(mix_EEG_col_test.shape)
# print(real_EEG_col_test.shape)
# np.save("./data/44780_mix_EEG.npy",mix_EEG_col)
# np.save("./data/44780_real_EEG.npy",real_EEG_col)
# 
# np.save("./data/11200_mix_EEG_test.npy",mix_EEG_col_test)
# np.save("./data/11200_real_EEG_test.npy",real_EEG_col_test)

mix_EEG_col=np.load('./data/44780_mix_EEG.npy')
real_EEG_col=np.load('./data/44780_real_EEG.npy')
mix_EEG,EEG=shuffle(mix_EEG_col,real_EEG_col)
train_mix_EEG=mix_EEG_col
std_train=np.std(train_mix_EEG, ddof = 1)
train_mix_EEG=train_mix_EEG/std_train

train_true_EEG=real_EEG_col
std_true=np.std(train_true_EEG, ddof = 1)
train_true_EEG=train_true_EEG/std_true


mix_EEG_col_test=np.load('./data/11200_mix_EEG_test.npy')
real_EEG_col_test=np.load('./data/11200_real_EEG_test.npy')

test_mix_EEG=mix_EEG_col_test
std_test=np.std(test_mix_EEG, ddof = 1)
test_mix_EEG=test_mix_EEG/std_test

test_true_EEG=real_EEG_col_test
std_true_test=np.std(test_true_EEG, ddof = 1)
test_true_EEG=test_true_EEG/std_true_test

#
#
# test_mix_EEG=mix_EEG[44780:50380,:]
# test_true_EEG=EEG[44780:50380,:]
#
# pre_mix_EEG=mix_EEG[50380:55980,:]
# pre_true_EEG=EEG[50380:55980,:]


dataset=customData(train_mix_EEG,train_true_EEG)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


dataset_test=customData(test_mix_EEG,test_true_EEG,)
dataloader_test= DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
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
dir_gen='./train/gen'
dir_disc='./train/disc'
epochpoint=0
if os.path.exists('./train/gen'):
    all = os.listdir(dir_gen)
    if len(all)>0:
        checkpoint=torch.load('./train/gen/'+all[len(all)-1])
        gen.load_state_dict(checkpoint['model'])
        # opt_gen.load_state_dict(checkpoint['optimizer'])
        epochpoint = checkpoint['epoch']
        print("gen模型加载{}".format(all[len(all)-1]))

if os.path.exists(dir_disc):
    all = os.listdir(dir_disc)
    if len(all)>0:
        checkpoint=torch.load('./train/disc/'+all[len(all)-1])
        disc.load_state_dict(checkpoint['model'])
        # opt_disc.load_state_dict(checkpoint['optimizer'])
        epochpoint = checkpoint['epoch']
        print("disc模型加载{}".format(all[len(all)-1]))

"""
############################################################
loss函数
############################################################
"""

criterion = nn.BCELoss()
criterion2=nn.MSELoss()




step = 0


loss_G=[]
loss_D=[]
loss_FR=[]
"""
############################################################
模型训练
############################################################
"""
for epoch in range(NUM_EPOCHS):

    for batch_idx, (real, _) in enumerate(tqdm(dataloader)):
        gen.train()
        disc.train()
        # print(batch_idx)
        real=real.view(-1,1,1024)
        _=_.view(-1,1,1024).to(device).float()
        # print(real.shape)
        real = real.to(device).float()
        # noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(real)
        fake=fake.view(-1,1,1024)
        # print(real.shape)
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(_).reshape(-1)
        # print(disc_real.shape)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        # print(fake.shape)
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        loss_real_fake_gen=criterion2(fake,_)

        gen.zero_grad()
        # loss_gen.backward(retain_graph=True)
        loss_GEN=(loss_gen+loss_real_fake_gen)/2
        loss_GEN.backward()
        opt_gen.step()
        loss_G.append(loss_gen.item())
        loss_D.append(loss_disc.item())
        loss_FR.append(loss_real_fake_gen.item())



    # Print losses occasionally and print to tensorboard
        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {np.mean(loss_D):.4f}, loss G: {np.mean(loss_G):.4f}, loss G_R: {np.mean(loss_FR):.4f}"
            )
            step += 1


    with torch.no_grad():
        """
        ############################################################
            tensorboard绘制三个loss计算曲线
        ############################################################
        """
        gen.eval()
        next,real=test(gen,dataloader_test)
        next=next.cpu().numpy()
        real=real.cpu().numpy()
        next=next*std_true_test
        real=real*std_true_test
        # print(next.shape)
        RRMSE_t,RRMSE_t_max,RRMSE_t_min=get_RRMSE_t(next,real)
        # print("RRMSE_t avg:{} max:{} min:{}".format(RRMSE_t,RRMSE_t_max,RRMSE_t_min))
        CC,CC_max,CC_min=get_CC(next,real)
        # print("CC avg:{} max:{} min:{}".format(CC,CC_max,CC_min))
        RRMSE_f,RRMSE_f_max,RRMSE_f_min=getPSD(next,real)
        # print("psd RRMSE_f avg:{} max:{} min:{}".format(RRMSE_f,RRMSE_f_max,RRMSE_f_min))

        niter = epoch
        writer_real = SummaryWriter(f"logs")
        writer_real.add_scalars('Loss/Train_D_loss', {'val_loss': np.mean(loss_D)}, niter)
        writer_real.add_scalars('Loss/Train_G_loss', {'val_loss': np.mean(loss_G)}, niter)
        writer_real.add_scalars('Loss/Train_G_R_loss', {'val_loss': np.mean(loss_FR)}, niter)
        writer_real.add_scalars('test/RRMSE_t-psd-CC',{'RRMSE_t':RRMSE_t,'RRMSE_f':RRMSE_f,'CC':CC},niter)
        writer_real.close()
        loss_G=[]
        loss_D=[]
        loss_FR=[]

    """
############################################################
模型保存
############################################################
    """
    if epoch%5==0:
        if epochpoint<=(epoch/5):
            epochpoint=epoch/5
        else:
            epochpoint=epochpoint+1
        state_gen = {'model':gen.state_dict(), 'optimizer':opt_gen.state_dict(), 'epoch':epochpoint}
        state_disc = {'model':disc.state_dict(), 'optimizer':opt_disc.state_dict(), 'epoch':epochpoint}
        torch.save(state_gen,'./train/gen/epoch_{}_gen.pth'.format(epochpoint))
        torch.save(state_disc,'./train/disc/epoch_{}_disc.pth'.format(epochpoint))
        print("模型保存{}_epoch".format(epochpoint))