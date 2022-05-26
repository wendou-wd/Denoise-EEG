import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft

import matplotlib.pylab as pylab
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
    # fig, ax = plt.subplots(3, 2,figsize=(6, 6))
    # fig.subplots_adjust(hspace=0.4)
    # ax[0][0].plot( EEG_col_test[0,:],"y",label="EEG",linewidth=0.5)
    # ax[0][0].set_title('Amplitude(uV)')
    # ax[0][1].set_title('PSD(dB)')
    # ax[0][0].legend()
    # ax[0][1].psd(EEG_col_test[0,:],1024,1024,color="y",linewidth=0.5)
    # ax[1][0].plot( EMG[1,:],"g",label="EMG",linewidth=0.5)
    # ax[1][0].legend()
    # ax[1][1].psd(EMG[1,:],1024,1024,color="g",linewidth=0.5)
    # ax[2][0].plot( EOG[0,:],"b",label="EOG",linewidth=0.5)
    # ax[2][0].legend()
    # ax[2][1].psd(EOG[0,:],1024,1024,color="b",linewidth=0.5)
    # plt.show()
    # fig.savefig("sample.tiff",dpi=600)

    params = {
        'axes.labelsize': '16',
        'xtick.labelsize': '16',
        'ytick.labelsize': '16',
        'lines.linewidth': '2',
        'legend.fontsize': '20',
        'figure.figsize': '26, 24'  # set figure size
    }
    pylab.rcParams.update(params)
    mix_EEG_EMG=np.load('./data/44780_mix_EEG.npy')
    true_EEG_EMG=np.load('./data/44780_real_EEG.npy')
    mix_EEG_EOG=np.load('./data/mix_EOG_27200.npy')
    true_EEG_EOG=np.load('./data/true_27200.npy')
    fig, ax = plt.subplots(1, 2,figsize=(12,6))
    fig.subplots_adjust(wspace=0.3)
    # plt.xticks(fontname="Times New Roman", fontsize=25)
    # plt.yticks(fontname="Times New Roman", fontsize=25)
    # ax[0].set_xticks(fontsize=25)
    ax[0].plot( mix_EEG_EMG[0,:],"gray",label="NoiseEEG-EMG")
    ax[0].plot( true_EEG_EMG[0,:],"black",label="EEG")
    ax[0].legend(loc='upper right', fontsize=18)

    ax[0].set_ylabel('Amplitude(uV)',fontsize=18)
    ax[0].set_title('a', fontsize=25)
    ax[1].plot(  mix_EEG_EOG[0,:],"gray",label="NoiseEEG-EOG",linewidth=1)
    ax[1].plot( true_EEG_EOG[0,:],"black",label="EEG",linewidth=1)
    ax[1].legend(loc='upper right', fontsize=18)
    ax[1].set_title('b', fontsize=25)
    plt.show()
    fig.savefig("sample_mix.tiff")
    # D=[0.797605574,
    #    0.702069044,
    #    0.697834849,
    #    0.696248889,
    #    0.695547342,
    #    0.694972932,
    #    0.694570065,
    #    0.694304347,
    #    0.69411546,
    #    0.69399482,
    #    0.693906486,
    #    0.693816781,
    #    0.693751872,
    #    0.693696678,
    #    0.693649888,
    #    0.693617105,
    #    0.693578959,
    #    0.693567455,
    #    0.693528771,
    #    0.693512261,
    #    0.693505466,
    #    0.693476737,
    #    0.693474233,
    #    0.693459988,
    #    0.693438053,
    #    ]
    #
    # G=[0.610510111,
    #    0.683720469,
    #    0.686344206,
    #    0.688427389,
    #    0.690896153,
    #    0.691642642,
    #    0.692173421,
    #    0.692538083,
    #    0.692548096,
    #    0.692695439,
    #    0.693050265,
    #    0.693169475,
    #    0.693237841,
    #    0.693175316,
    #    0.693166256,
    #    0.693158388,
    #    0.693096161,
    #    0.69311893,
    #    0.693062544,
    #    0.693148613,
    #    0.693205893,
    #    0.693158865,
    #    0.693180323,
    #    0.693123877,
    #    0.693256855
    #    ]
    # mse=[0.557266355,
    #      0.296731323,
    #      0.246875435,
    #      0.210494548,
    #      0.174015552,
    #      0.148122177,
    #      0.128575906,
    #      0.112969771,
    #      0.100866705,
    #      0.090634309,
    #      0.082432508,
    #      0.076288819,
    #      0.070150867,
    #      0.06520588,
    #      0.06019799,
    #      0.056555524,
    #      0.052755568,
    #      0.050079815,
    #      0.047359467,
    #      0.044981994,
    #      0.042130802,
    #      0.040720753,
    #      0.038584191,
    #      0.036979619,
    #      0.035802949
    #      ]
    # t=[0.588456452,
    #    0.542641461,
    #    0.464067042,
    #    0.445970207,
    #    0.426555485,
    #    0.390181988,
    #    0.381644934,
    #    0.374739647,
    #    0.368138701,
    #    0.353637755,
    #    0.357609719,
    #    0.348033637,
    #    0.329147786,
    #    0.326909453,
    #    0.338523954,
    #    0.320652813,
    #    0.310791671,
    #    0.310704172,
    #    0.304921478,
    #    0.303671658,
    #    0.312530935,
    #    0.308348268,
    #    0.295126915,
    #    0.298800498,
    #    0.291999727
    #    ]
    # f=[0.516228378,
    #    0.61333245,
    #    0.412765592,
    #    0.425955206,
    #    0.410588771,
    #    0.36243847,
    #    0.339106143,
    #    0.339526325,
    #    0.330783725,
    #    0.329950899,
    #    0.339500725,
    #    0.333191991,
    #    0.297089726,
    #    0.288409978,
    #    0.312794834,
    #    0.291430324,
    #    0.280675381,
    #    0.285615712,
    #    0.277710795,
    #    0.276566029,
    #    0.297742724,
    #    0.296107292,
    #    0.271462351,
    #    0.28066206,
    #    0.268200666
    #    ]
    # c=[0.808224022,
    #    0.842421114,
    #    0.874920249,
    #    0.883081496,
    #    0.893729925,
    #    0.907702148,
    #    0.910639465,
    #    0.913737178,
    #    0.916989505,
    #    0.922725797,
    #    0.922176719,
    #    0.926910341,
    #    0.932583869,
    #    0.933332264,
    #    0.928703189,
    #    0.935535669,
    #    0.93910861,
    #    0.938577771,
    #    0.940596282,
    #    0.941170871,
    #    0.938890755,
    #    0.942342341,
    #    0.943803012,
    #    0.943013012,
    #    0.944992661
    #    ]
    # fig, ax = plt.subplots(2, 2)
    # ax[0][0].plot(D,label="Loss(D)")
    # # ax[0][0].set_title('Amplitude(uV)')
    # # ax[0][1].set_title('PSD(dB)')
    # ax[0][0].legend(loc='upper right')
    # ax[0][0].set_title("a")
    # ax[0][1].set_title("b")
    # ax[1][0].set_title("c")
    # ax[1][1].set_title("d")
    # ax[0][1].plot(G,label="Loss(G)")
    # ax[0][1].legend(loc='lower right')
    # ax[1][0].plot(mse,label="MSE")
    # ax[1][0].legend(loc='upper right')
    # ax[1][1].plot(t,label="RRMSET")
    # ax[1][1].plot(f,label="RRMSES")
    # ax[1][1].plot(c,label="CC")
    # ax[1][1].legend(loc='right')
    # plt.show()
    # fig.savefig("MSE.tiff",dpi=600)

