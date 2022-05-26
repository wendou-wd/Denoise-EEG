import scipy.io as sio
import numpy as np

import random
from torch.utils.data import DataLoader, Dataset
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