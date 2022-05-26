import matplotlib.pyplot as plt
import matplotlib
import numpy as np
if __name__ == '__main__':

    # 设置中文字体和负号正常显示
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    label_list = ['Filter', 'HHT', 'EMD', 'FCNN','SimpleCNN','ComplexCNN','RNN','NovelCNN','  DeepSeparator','EEGD','GAN-LSTM','GAN-1D-CNN']    # 横坐标刻度显示值
    # num_list1 = [1.113,1.718,1.094,0.585,0.646,0.650,0.570,0.448,0.712,0.677,0.292]      # 纵坐标值1
    # num_list2 = [1.237,1.705,2.333,0.580,0.649,0.633,0.530,0.442,0.717,0.626,0.268]      # 纵坐标值2
    num_list3= [0.443,0.527,0.672,0.796,0.783,0.780,0.812,0.863,0.734,0.732,0.650,0.945]
    # x = range(len(num_list1))
    bar_width = 0.6
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """
    rects1 = plt.bar(np.arange(12), height=num_list3, width=0.5, color='gray', label="CC")
    # rects2 = plt.bar(np.arange(11)+bar_width, height=num_list2, width=0.3, color='#DDDDDD', label="RRMSES")
    # rects3 = plt.bar(np.arange(11)+bar_width*2, height=num_list3, width=0.3, color='#414141', label="CC")
    plt.ylim(0, 1.2)     # y轴取值范围
    # plt.ylabel("数量")
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    # """
    plt.xticks(np.arange(12), label_list)
    # plt.xlabel("年份")
    plt.title("myogenic artifact removal",fontsize=20)
    plt.legend()     # 设置题注
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2-0.01, height+0.01, str(height), ha="center", va="bottom",fontsize=10)
    # for rect in rects2:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2+0.01, height+0.01, str(height), ha="center", va="bottom",fontsize=8)
    # for rect in rects3:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height+0.1, str(height), ha="center", va="bottom")
    fig = plt.gcf()
    plt.show()
    fig.savefig('cc.tiff', dpi=600)