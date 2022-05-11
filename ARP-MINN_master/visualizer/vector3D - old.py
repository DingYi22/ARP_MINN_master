# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:54:44 2020

@author: ustarlee

"""

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D

class vector3D(object):
    def __init__(self, n_components = 3):
        self.n_components = n_components # 降维维数
        
    def draw_and_save(self, feature, label, name=None, save_path=None):
        # data process
        x = np.zeros([feature.shape[0], 2])
        y = np.zeros([feature.shape[0], 2])
        z = np.zeros([feature.shape[0], 2])
        x[:,1], y[:,1], z[:,1] = feature[:,0], feature[:,1], feature[:,2]
        plt.close('all')
        style = {'family': 'Times New Roman', 
                  'weight': 'normal', 
                  'color':  'black', 
                  'size': 15
                }
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(8, 6), dpi=300)
        # ax = Axes3D(fig)
        ax = fig.gca(projection='3d')
        # ax = fig.add_subplot(111, projection='3d')
        
        # for i in range(x.shape[0]):
        #     color = 'blue' if label[i] ==  1. else 'red'
        #     # len_quiver = (x[i,1]**2 + y[i,1]**2 + z[i,1]**2)**0.5
        #     # ax.quiver(x[i,0], y[i,0], z[i,0], x[i,1], y[i,1], z[i,1], color=color, length=len_quiver, arrow_length_ratio=0.2,normalize=True) 
        #     ax.plot(x[i,:], y[i,:], z[i,:], color=color, linewidth='2')
        #     if name:
        #         ax.text(x[i,1], y[i,1], z[i,1], name[i], **style)
        
        for i in range(x.shape[0]):
            if label[i] == 1.:
                l1 = ax.plot(x[i,:], y[i,:], z[i,:], color='red', linewidth='2', label='positive instance')
            else:
                l2 = ax.plot(x[i,:], y[i,:], z[i,:], color='lightskyblue', linewidth='2', label='negative instance')
            if name:
                ax.text(x[i,1], y[i,1], z[i,1], name[i], **style)
        
        # ax.tick_params(axis='both',which='major',labelsize=15)
        ax.set(xlabel='X', ylabel='Y', zlabel='Z')
        # ax.grid(b=True)
        # color = ['red', 'lightskyblue']
        # labels = ['positive instance', 'negative instance']
        # patches = [ mpatches.Patch(color=color[i], linestyle='-', label="{:.}".format(labels[i]) ) for i in range(len(color)) ]
        # ax.legend(handles=patches)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        # ax.view_init(elev=20, azim=30)   # 仰角，方位角
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0., dpi=300)
        
        plt.show()
        plt.close()
        
    def visualizer(self, feature, label, save_path = None, name=None):
        """
        bag: numpy array [num, dim]
        label: list of label
        """
        pca = PCA(n_components=self.n_components)     # 实例化PCA
        pca.fit(feature)                              # 训练
        feature_pca = pca.fit_transform(feature)   # 降维结果
        # print('Contribution rate:', pca.explained_variance_ratio_)         # 输出贡献率
        if self.n_components == 3:
            self.draw_and_save(feature_pca, label, name, save_path)
        
        return feature_pca


if __name__ == '__main__':
    # 反包
    bag0 = torch.tensor([[4.0087e-01, 0.0000e+00, 3.0904e-01, 3.3557e-01, 5.1772e-01, 8.7555e-02,
             4.9042e-01, 2.5154e-01, 0.0000e+00, 1.9718e-01, 2.8118e-02, 0.0000e+00,
             5.4869e-01, 0.0000e+00, 3.7404e-01, 0.0000e+00, 7.3966e-01, 1.6676e-02,
             0.0000e+00, 0.0000e+00, 6.2635e-01, 0.0000e+00, 2.5769e-01, 5.2086e-01,
             5.9850e-02, 4.8377e-04, 2.0165e-01, 4.7579e-01, 2.5111e-01, 0.0000e+00,
             0.0000e+00, 5.6240e-01, 5.6831e-03, 0.0000e+00, 0.0000e+00, 1.0488e-02,
             1.1650e-01, 8.2224e-03, 1.3778e-03, 2.7320e-01, 0.0000e+00, 7.5146e-04,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 4.1902e-01, 0.0000e+00, 2.0498e-01,
             6.0394e-02, 1.7305e-01, 1.8245e-01, 4.9227e-01, 5.5360e-02, 0.0000e+00,
             3.0393e-01, 0.0000e+00, 0.0000e+00, 7.5155e-01, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 8.6407e-01, 1.1104e+00, 0.0000e+00]])
    # 反包
    bag1 = torch.tensor([[0.3113, 0.0208, 0.2751, 0.2321, 0.4682, 0.1053, 0.3942, 0.1799, 0.0000,
             0.1534, 0.0000, 0.0381, 0.4421, 0.0000, 0.2663, 0.0000, 0.5430, 0.0262,
             0.0000, 0.0000, 0.4679, 0.0119, 0.1600, 0.3138, 0.0000, 0.0000, 0.1470,
             0.3225, 0.1905, 0.0000, 0.0000, 0.4763, 0.0127, 0.0000, 0.0000, 0.0533,
             0.1232, 0.1064, 0.0650, 0.2370, 0.0641, 0.0329, 0.0000, 0.0000, 0.0000,
             0.3504, 0.0239, 0.1274, 0.0966, 0.1485, 0.1627, 0.3699, 0.0508, 0.0000,
             0.2298, 0.0000, 0.0000, 0.5850, 0.0000, 0.0000, 0.0000, 0.6911, 0.8540,
             0.0000]])
    # 反包
    bag2 = torch.tensor([[0.2773, 0.0236, 0.2578, 0.3048, 0.4409, 0.1082, 0.3410, 0.1954, 0.0000,
             0.1739, 0.0000, 0.0579, 0.5458, 0.0041, 0.3485, 0.0414, 0.6566, 0.0403,
             0.0000, 0.0000, 0.5719, 0.0035, 0.2138, 0.4488, 0.1286, 0.0000, 0.1669,
             0.4225, 0.2158, 0.0000, 0.0000, 0.4985, 0.0676, 0.0000, 0.0000, 0.0958,
             0.0772, 0.0475, 0.0472, 0.1965, 0.0474, 0.0507, 0.0000, 0.0000, 0.0676,
             0.3652, 0.0885, 0.2389, 0.0804, 0.1269, 0.1146, 0.4329, 0.0854, 0.0000,
             0.2507, 0.0000, 0.1247, 0.6378, 0.0159, 0.0000, 0.0000, 0.6950, 0.8934,
             0.0377]])
    # 正包
    bag3 = torch.tensor([[2.9018e-02, 2.0909e-01, 2.1207e-03, 1.0858e-02, 9.6362e-03, 3.2391e-01,
             5.6668e-03, 3.2615e-02, 0.0000e+00, 1.1249e-03, 0.0000e+00, 5.0892e-01,
             9.0273e-03, 9.9529e-03, 1.3487e-02, 0.0000e+00, 4.2797e-03, 6.1576e-02,
             0.0000e+00, 1.3994e-03, 1.4064e-02, 5.5426e-02, 1.2477e-01, 3.2112e-03,
             0.0000e+00, 0.0000e+00, 1.1875e-02, 1.4593e-02, 1.0036e-02, 0.0000e+00,
             0.0000e+00, 4.8388e-03, 4.5716e-01, 0.0000e+00, 0.0000e+00, 4.7328e-01,
             6.4514e-02, 5.4355e-01, 4.7775e-01, 5.4476e-02, 5.4583e-01, 2.0456e-01,
             0.0000e+00, 1.3017e-02, 2.2210e-01, 9.7394e-02, 4.9541e-01, 3.3838e-02,
             3.0328e-01, 1.0870e-02, 5.1761e-02, 9.3407e-03, 0.0000e+00, 0.0000e+00,
             1.4629e-04, 0.0000e+00, 0.0000e+00, 1.4755e-02, 1.6936e-01, 1.6184e-01,
             0.0000e+00, 1.0070e-02, 1.7137e-02, 2.8690e-01]])
    # 反包
    bag4 = torch.tensor([[0.2621, 0.0745, 0.2579, 0.1407, 0.4029, 0.1211, 0.2914, 0.1715, 0.0000,
             0.1027, 0.0000, 0.1401, 0.3975, 0.0211, 0.3252, 0.0375, 0.3265, 0.0626,
             0.0000, 0.0000, 0.3682, 0.0251, 0.1235, 0.2479, 0.0000, 0.0000, 0.1256,
             0.2720, 0.1486, 0.0000, 0.0288, 0.3713, 0.0890, 0.0000, 0.0000, 0.1038,
             0.1182, 0.1698, 0.1157, 0.1531, 0.1731, 0.1109, 0.0000, 0.0000, 0.0161,
             0.2482, 0.0761, 0.1757, 0.0981, 0.1305, 0.0218, 0.3065, 0.0363, 0.0000,
             0.1643, 0.0000, 0.0434, 0.4720, 0.0320, 0.0072, 0.0000, 0.5236, 0.6280,
             0.0345]])
    # 反包
    bag5 = torch.tensor([[0.4607, 0.0000, 0.3651, 0.3666, 0.6760, 0.0098, 0.5314, 0.2757, 0.0000,
             0.2838, 0.0194, 0.0000, 0.6855, 0.0000, 0.5050, 0.0178, 0.9104, 0.0368,
             0.0000, 0.0000, 0.7891, 0.0000, 0.2454, 0.6496, 0.1066, 0.0000, 0.2352,
             0.6085, 0.3097, 0.0000, 0.0000, 0.6299, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0691, 0.0000, 0.0000, 0.2946, 0.0000, 0.0107, 0.0000, 0.0000, 0.0060,
             0.4534, 0.0000, 0.3006, 0.0000, 0.1877, 0.1178, 0.6263, 0.1152, 0.0000,
             0.3975, 0.0000, 0.1600, 0.9034, 0.0000, 0.0000, 0.0000, 0.9994, 1.2486,
             0.0000]])
    # 正包
    bag6 = torch.tensor([[0.0231, 0.0000, 0.0000, 0.1806, 0.3146, 0.1382, 0.3681, 0.1997, 0.1134,
             0.1317, 0.1655, 0.0964, 0.0292, 0.0725, 0.0870, 0.1254, 0.0000, 0.0749,
             0.0000, 0.1459, 0.2657, 0.0202, 0.0000, 0.2516, 0.0397, 0.0007, 0.1582,
             0.1651, 0.0000, 0.0000, 0.0000, 0.2836, 0.3690, 0.0373, 0.2876, 0.0795,
             0.0264, 0.0000, 0.1773, 0.2179, 0.3322, 0.3389, 0.1929, 0.1169, 0.0792,
             0.1537, 0.0000, 0.0308, 0.0000, 0.2063, 0.0954, 0.3223, 0.0898, 0.0000,
             0.1951, 0.0000, 0.2269, 0.0753, 0.1032, 0.1149, 0.1680, 0.0000, 0.0207,
             0.4227]])
    # 正包
    bag7 = torch.tensor([[0.2223, 0.0000, 0.0000, 0.4587, 0.6285, 0.0411, 0.6962, 0.1789, 0.0951,
             0.1758, 0.1211, 0.2082, 0.0000, 0.2031, 0.2795, 0.2995, 0.0000, 0.0826,
             0.0000, 0.1130, 0.4594, 0.0431, 0.0000, 0.4592, 0.0000, 0.0000, 0.4102,
             0.5179, 0.0000, 0.0000, 0.0000, 0.5628, 0.7606, 0.0000, 0.5138, 0.0597,
             0.1287, 0.0000, 0.0398, 0.1091, 0.7005, 0.6661, 0.5075, 0.1127, 0.1419,
             0.2511, 0.0000, 0.1006, 0.0379, 0.3712, 0.0951, 0.8478, 0.3006, 0.0000,
             0.0687, 0.0000, 0.5893, 0.0522, 0.1668, 0.0972, 0.1076, 0.0000, 0.0303,
             0.9019]])

    bag = torch.cat((bag3, bag6, bag7, bag0, bag1, bag2, bag4, bag5), 0)
    
    dist = torch.ones(bag.shape[0], bag.shape[0])
    for i in range(bag.shape[0]):
        for j in range(bag.shape[0]):
            dist[i,j] = torch.norm((bag[i]- bag[j]), -1)
    
    print(dist)
    
    similarity = torch.matmul(bag, bag.transpose(1,0))
    print(similarity)

    bag = bag.numpy()
    label = [1., 1., 1., 0., 0., 0., 0., 0.]
    name = ['bag3', 'bag6', 'bag7', 'bag0', 'bag1', 'bag2', 'bag4', 'bag5']
    save_path = '../result/vector3D.png'
    vector3D = vector3D(3).visualizer(bag, label, save_path, name)







