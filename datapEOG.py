import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft
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
    return  10 * np.log10(np.abs(pos_Y_from_shift)**2/1024+0.1)
if __name__ == '__main__':
    # EEG_col_test=np.load('./data/true_test_6800.npy')
    # EMG=np.load('./data/EMG_1024.npy')
    # EOG=np.load('./data/EOG_3400_1024.npy')
    # fig, ax = plt.subplots(3, 2)
    # ax[0][0].plot( EEG_col_test[0,:],"y",label="EEG")
    # ax[0][0].set_title('Amplitude(uV)')
    # ax[0][1].set_title('PSD(dB)')
    # ax[0][0].legend()
    # ax[0][1].psd(EEG_col_test[0,:],1024,1024,color="y")
    # ax[1][0].plot( EMG[1,:],"g",label="EMG")
    # ax[1][0].legend()
    # ax[1][1].psd(EMG[1,:],1024,1024,color="g")
    # ax[2][0].plot( EOG[0,:],"b",label="EOG")
    # ax[2][0].legend()
    # ax[2][1].psd(EOG[0,:],1024,1024,color="b")
    # plt.show()

    # mix_EEG_EMG=np.load('./data/44780_mix_EEG.npy')
    # true_EEG_EMG=np.load('./data/44780_real_EEG.npy')
    # mix_EEG_EOG=np.load('./data/mix_EOG_27200.npy')
    # true_EEG_EOG=np.load('./data/true_27200.npy')
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot( mix_EEG_EMG[0,:],"g",label="NoiseEEG-EMG")
    # ax[0].plot( true_EEG_EMG[0,:],"r",label="EEG")
    # ax[0].legend(loc='upper right', fontsize=10)
    #
    # ax[0].set_ylabel('Amplitude(uV)')
    # ax[0].set_title('a')
    # ax[1].plot(  mix_EEG_EOG[0,:],"y",label="NoiseEEG-EOG")
    # ax[1].plot( true_EEG_EOG[0,:],"r",label="EEG")
    # ax[1].legend(loc='upper right', fontsize=10)
    # ax[1].set_title('b')
    # plt.show()
    D=[0.856249034,
       0.710885525,
       0.70453006,
       0.700759113,
       0.698665917,
       0.697271287,
       0.696651936,
       0.695761204,
       0.6949808,
       0.693784952,
       0.694098771,
       0.694009602,
       0.693944633,
       0.693810403,
       0.693717718,
       0.693642616,
       0.69360882,
       0.693556488,
       0.693511903,
       0.693478882,
       0.693486512,
       0.693431318,
       0.693415463,
       0.693409204,
       0.69339186,
       0.6933797

       ]

    G=[0.544584095,
       0.680278361,
       0.682533145,
       0.685653031,
       0.686979473,
       0.688038945,
       0.689399064,
       0.690206289,
       0.690720916,
       0.683746576,
       0.6934129,
       0.691832662,
       0.692563832,
       0.693556011,
       0.691681385,
       0.694002211,
       0.69318974,
       0.693255067,
       0.693403661,
       0.693306208,
       0.693252861,
       0.693086028,
       0.693349659,
       0.693174958,
       0.693261445,
       0.693142056

       ]
    mse=[0.731941283,
         0.232309148,
         0.142687201,
         0.111462861,
         0.089115165,
         0.076021865,
         0.045936782,
         0.033866949,
         0.030993283,
         0.292727768,
         0.057451535,
         0.06483791,
         0.043095313,
         0.041227151,
         0.058382288,
         0.040650051,
         0.038797565,
         0.034722365,
         0.02887788,
         0.023639716,
         0.023416791,
         0.021593178,
         0.020871136,
         0.019531721,
         0.019650994,
         0.018989956

         ]
    t=[0.646534681,
       0.54282409,
       0.504330099,
       0.616421819,
       0.4737041,
       0.446580499,
       0.43769142,
       0.436555833,
       0.420387596,
       0.469536722,
       0.437637448,
       0.41932565,
       0.415027767,
       0.463555127,
       0.440930277,
       0.437673628,
       0.416031092,
       0.430635661,
       0.41618368,
       0.421559662,
       0.413056731,
       0.411642104,
       0.409687787,
       0.416246384,
       0.408977747,
       0.408970267,

       ]
    f=[0.523390234,
       0.490683645,
       0.465379655,
       0.663486779,
       0.433272809,
       0.417755663,
       0.424010009,
       0.43586272,
       0.396538168,
       0.426903665,
       0.399540484,
       0.391235113,
       0.391384363,
       0.467636257,
       0.426874727,
       0.432046294,
       0.394706428,
       0.4140957,
       0.397413552,
       0.421993673,
       0.403763175,
       0.414754242,
       0.398844153,
       0.415460706,
       0.410145819,
       0.411051512

       ]
    c=[0.75230372,
       0.830973506,
       0.851516008,
       0.79662776,
       0.868144512,
       0.881907582,
       0.885966957,
       0.886296928,
       0.893249989,
       0.872000933,
       0.885565937,
       0.893513322,
       0.895227313,
       0.873273969,
       0.882887781,
       0.884953022,
       0.894341767,
       0.888786137,
       0.894234061,
       0.893363118,
       0.895995557,
       0.898582935,
       0.898486316,
       0.895669103,
       0.899629653,
       0.899731934
       ]
    fig, ax = plt.subplots(2, 2)
    ax[0][0].plot(D,label="Loss(D)")
    # ax[0][0].set_title('Amplitude(uV)')
    # ax[0][1].set_title('PSD(dB)')
    ax[0][0].legend(loc='upper right')
    ax[0][0].set_title("a")
    ax[0][1].set_title("b")
    ax[1][0].set_title("c")
    ax[1][1].set_title("d")
    ax[0][1].plot(G,label="Loss(G)")
    ax[0][1].legend(loc='lower right')
    ax[1][0].plot(mse,label="MSE")
    ax[1][0].legend(loc='upper right')
    ax[1][1].plot(t,label="RRMSET")
    ax[1][1].plot(f,label="RRMSES")
    ax[1][1].plot(c,label="CC")
    ax[1][1].legend(loc='right')
    plt.show()
    fig.savefig("MSE_eog.tiff",dpi=600)

