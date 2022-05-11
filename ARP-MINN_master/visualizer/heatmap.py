# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:33:28 2020

@author: ustarlee
"""
import re
import math
import torch
import imageio
import numpy as np
import scipy.io as sio
import matplotlib as mpl
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

class heatmap(object):
    def __init__(self, readpath, savepath):
        self.readpath = readpath + 'OriginalPicture/'
        self.savepath = savepath
        self.patch_size = np.array([27, 27]).astype(np.uint16)
        data2 = sio.loadmat(self.readpath + 'colon_cancer.mat')
        self.labels = data2['x']['label'][0, 0]
        self.positions = data2['x']['position'][0, 0]
        self.names = data2['x']['info_name'][0,0]
        
        
    def draw_heatmap(self, img, position, weight):
        """
        function:实现基于图像的注意力图的可视化
        思路： 对图像img按patchs块进行可视化，权重越大，亮度越高。权重矩阵需预处理， $a_k^' = (a_k - min(a)) / (max(a) - min(a))$
        input：
            img:输入图像，对该图像进行可视化操作  numpy [col, row, 3]
            position: patchs的位置信息          numpy [n, 2] 2维矩阵, 给的是中心点坐标
            patch_size: patchs的大小信息            numpy [ , ]
            weight: patchs对应的权重信息        numpy [n] 向量
            
            mask  [col, row]
        """
        # 预处理
        img = img.astype(np.float64)
        position = position.astype(np.uint16)
        weight = weight.astype(np.float64)
      
        wei_min, wei_max = min(weight), max(weight)
        if wei_max > wei_min:
            weight = (weight - wei_min) / (wei_max - wei_min)
        elif wei_max == wei_min:     # 若最大值==最小值， 则weight为同一个数构成的矩阵
            weight = weight / (wei_max)
        
        col, row, _ = img.shape
        mask = np.zeros([col, row])
        
        for i in range(position.shape[0]):
            pos = [position[i,0], position[i,1]]
            if weight[i] > 0. :
                mask = self.Patch_Padding(mask, pos, weight[i])
        
        mask_expand = np.tile(mask[:,:,np.newaxis], (1, 1, 3))  # [col, row] --> [col, row, 3]
        img_re = np.multiply(img, mask_expand)  # 逐点乘
        img_re = img_re.astype(np.uint16)
        
        return img_re
        
    
    def show_and_save(self, img, title=None, save_path=None):
        plt.close('all')
        style = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'color':  'black', 
                  'size': 8
                }
        
        plt.figure(dpi = 300)
        # ax = plt.gca()                # 原点左上角
        # ax.xaxis.set_ticks_position('top')
        plt.axis('off')  #去掉坐标轴
        if title:
            plt.title(title, **style)
        plt.imshow(img)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0., dpi=300)
        # plt.show()
        plt.close()
        
    
    def Patch_Padding(self, matrix, pos, value):
        """
        function: 将给定矩阵的特定区域赋值(同一值)
        区域: [ pos[0]:pos[0]+patch_size[0]-1, pos[1]:pos[1]+patch_size[1]-1 ]
        """
        pos = pos - (self.patch_size-1)//2   # 由中心点坐标计算左上角坐标
        col, row = matrix.shape
        for i in range(self.patch_size[0] * self.patch_size[1]):
            x, y = pos[0] + i//self.patch_size[0], pos[1] + i%self.patch_size[1]
            if x < col and y < row and matrix[x, y] < value:  # 防止暗色覆盖亮色
                matrix[x, y] = value
        
        return matrix
        
        
    def computer_gt_weight(self, position, gt):
        """
        function:根据ground truth标注信息 和 patchs块位置信息，标注注意力图
        input: 
            position 是patchs的位置信息 numpy [n,2]
            gt: 是该图像的ground truth 位置信息 [n,2]
        """
        weight = np.zeros([position.shape[0]])
        for i in range(position.shape[0]):
            pos_y, pos_x = position[i,0], position[i,1]
            for j in range(gt.shape[0]):
                gt_x, gt_y = gt[j,0], gt[j,1]
                
                if pos_x <= gt_x <= (pos_x+self.patch_size[0]-1) and pos_y <= gt_y <= (pos_y+self.patch_size[1]-1):
                    weight[i] = 1
    
        return weight
    
    
    def visualizer(self, img_name, weight, model_name):
        No = int(re.findall(r'\d+', img_name)[0])
        label = self.labels[No-1][0]
        position = self.positions[No-1, 0]
        gt = (sio.loadmat(self.readpath + img_name +'_epithelial.mat'))['detection']
        img = imageio.imread(self.readpath + img_name + '.bmp')

        if label == 0.:
            print("The label of input image is FALSE ! ")
        # 1. 绘制原始图像
        # title_ori = 'Original H&E stained histology image [' + img_name + ']'
        title_ori = None
        save_path_ori = self.savepath + img_name + "_Original.png"
        self.show_and_save(img, title_ori, save_path_ori)
    
        # 2. 绘制27*27patchs centered around all marked nuclei
        weight_pt = np.ones([position.shape[0]])
        # title_pt = '27*27 processed patchs [' + img_name + ']'
        title_pt = None
        save_path_pt = self.savepath + img_name + "_Patchs.png"
        img_re = self.draw_heatmap(img, position, weight_pt)
        self.show_and_save(img_re, title_pt, save_path_pt)
    
    
        # 3. 绘制ground truth
        weight_gt = self.computer_gt_weight(position, gt)
        # title_gt = 'Ground Truth: patchs [' + img_name+ ']'
        title_gt = None
        save_path_gt =self.savepath + img_name + "_GroundTruth.png"
        img_re = self.draw_heatmap(img, position, weight_gt)
        self.show_and_save(img_re, title_gt, save_path_gt)
    
        # 4. attention map
        
        # title_att = 'Attention heatmap [' + img_name+ ']'
        title_att = None
        save_path_att = self.savepath + img_name + '_' + model_name +".png"
        img_re = self.draw_heatmap(img, position, weight)
        self.show_and_save(img_re, title_att, save_path_att)
        
        return weight


if __name__ == '__main__':
    
    # 100张图像，只有正包的才有可视化的价值，确保图像为正包

    img_name = 'img80'
    weight = torch.tensor([[9.4130e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.4093e-03,
         2.8389e-04, 1.8630e-05, 0.0000e+00, 8.1599e-03, 0.0000e+00, 1.5853e-02,
         4.7587e-04, 2.5228e-04, 6.8410e-04, 5.9616e-03, 0.0000e+00, 1.3228e-03,
         0.0000e+00, 1.3858e-03, 1.9406e-03, 1.6762e-02, 1.4982e-04, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 2.8381e-03, 0.0000e+00, 6.9110e-03, 0.0000e+00,
         6.8085e-03, 0.0000e+00, 0.0000e+00, 8.5625e-05, 0.0000e+00, 1.4709e-02,
         0.0000e+00, 2.7957e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 3.4675e-03, 0.0000e+00, 3.3651e-03, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.2584e-03, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 7.4166e-03, 0.0000e+00, 0.0000e+00, 7.3708e-03,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.5396e-04, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 9.4699e-03, 4.8233e-03, 0.0000e+00, 1.1998e-02,
         8.5322e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.0613e-04,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 1.3979e-02, 4.1390e-03, 3.1912e-03, 4.8986e-03, 0.0000e+00,
         3.0758e-03, 7.6820e-03, 1.7770e-05, 9.0249e-03, 0.0000e+00, 2.4297e-03,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.0497e-04,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.5230e-03, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.9764e-03, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 1.6407e-03, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 3.4166e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         3.8634e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 2.7040e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         1.1786e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 4.1655e-05,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 4.4689e-03, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 1.0929e-02, 1.5106e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         4.9105e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 5.9624e-03, 4.8752e-03, 0.0000e+00, 0.0000e+00,
         2.4845e-03, 0.0000e+00, 8.7579e-04, 1.1995e-02, 0.0000e+00, 1.7374e-03,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 5.1948e-03, 6.6168e-03, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 1.7043e-03, 0.0000e+00, 0.0000e+00, 3.5158e-03,
         1.0391e-02, 0.0000e+00, 0.0000e+00, 1.2059e-02, 0.0000e+00, 0.0000e+00,
         8.6844e-03, 2.1871e-02, 0.0000e+00, 4.1604e-03, 8.2990e-03, 9.7828e-03,
         1.8330e-05, 1.4781e-02, 8.2287e-03, 1.0727e-02, 0.0000e+00, 4.6602e-03,
         0.0000e+00, 1.2906e-02, 0.0000e+00, 0.0000e+00, 8.2163e-03, 0.0000e+00,
         7.3108e-03, 0.0000e+00, 0.0000e+00, 2.7872e-03, 2.7385e-03, 0.0000e+00,
         9.5943e-03, 0.0000e+00, 7.1360e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 7.4425e-03, 0.0000e+00, 4.2545e-03, 0.0000e+00, 1.5743e-02,
         0.0000e+00, 8.8109e-03, 1.3059e-02, 1.4783e-03, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 6.7646e-03, 0.0000e+00, 6.3613e-03, 9.8478e-03,
         6.9198e-04, 3.9858e-03, 1.2670e-03, 0.0000e+00, 0.0000e+00, 1.6699e-04,
         1.1683e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.3596e-04, 0.0000e+00,
         8.8552e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 8.9742e-04, 0.0000e+00,
         0.0000e+00, 4.0760e-03, 0.0000e+00, 1.5423e-04, 6.9069e-03, 4.5802e-03,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         6.5084e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.3204e-03, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 2.6634e-03, 0.0000e+00, 5.2308e-03,
         0.0000e+00, 1.4592e-03, 4.1430e-04, 0.0000e+00, 2.9255e-03, 1.4786e-03,
         0.0000e+00, 2.0254e-02, 0.0000e+00, 2.2490e-04, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 7.8016e-03, 1.0058e-02, 2.1116e-04,
         0.0000e+00, 0.0000e+00, 1.1889e-02, 1.0929e-02, 0.0000e+00, 0.0000e+00,
         5.9100e-03, 1.2634e-02, 0.0000e+00, 3.7269e-03, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 5.6192e-03, 1.1406e-02, 0.0000e+00, 1.7341e-03,
         8.7834e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.1308e-03,
         7.5280e-03, 1.2875e-02, 0.0000e+00, 1.5237e-02, 0.0000e+00, 0.0000e+00,
         1.5209e-03, 0.0000e+00, 2.1055e-04, 6.1547e-03, 0.0000e+00, 7.7667e-04,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0425e-04, 0.0000e+00, 2.1729e-04,
         0.0000e+00, 0.0000e+00, 7.5364e-03, 0.0000e+00, 0.0000e+00, 1.0995e-02,
         0.0000e+00, 2.4497e-03, 5.1821e-03, 2.0360e-04, 4.4797e-03, 4.0388e-03,
         1.9336e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 1.0129e-02, 0.0000e+00, 9.3544e-03, 0.0000e+00,
         1.0331e-02, 1.1363e-02, 0.0000e+00, 3.3176e-03, 6.5714e-03, 0.0000e+00,
         1.7219e-02, 0.0000e+00, 9.4118e-03, 0.0000e+00, 9.6866e-03, 2.6636e-03,
         6.1006e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.7966e-02, 0.0000e+00,
         0.0000e+00, 1.5453e-04, 0.0000e+00, 1.2404e-02, 7.9145e-04, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.1351e-03, 0.0000e+00,
         0.0000e+00, 1.0448e-02, 0.0000e+00, 0.0000e+00, 3.3453e-03, 0.0000e+00,
         0.0000e+00, 1.1289e-04, 1.0324e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 2.2791e-04, 1.5820e-02, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 4.5089e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         2.5476e-03, 9.6544e-03, 0.0000e+00, 1.1038e-04, 0.0000e+00, 7.3765e-04,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         1.0189e-02, 3.7111e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.5346e-02,
         1.5462e-04]])
    # weight = torch.tensor([[0.0078, 0.0013, 0.0033, 0.0009, 0.0003, 0.0052, 0.0007, 0.0015, 0.0003,
    #      0.0049, 0.0004, 0.0080, 0.0032, 0.0018, 0.0031, 0.0015, 0.0003, 0.0026,
    #      0.0003, 0.0036, 0.0041, 0.0091, 0.0019, 0.0003, 0.0003, 0.0010, 0.0037,
    #      0.0003, 0.0065, 0.0003, 0.0045, 0.0003, 0.0004, 0.0003, 0.0003, 0.0091,
    #      0.0004, 0.0022, 0.0003, 0.0003, 0.0003, 0.0003, 0.0004, 0.0028, 0.0003,
    #      0.0052, 0.0003, 0.0008, 0.0006, 0.0003, 0.0003, 0.0003, 0.0012, 0.0003,
    #      0.0003, 0.0003, 0.0057, 0.0008, 0.0003, 0.0067, 0.0003, 0.0003, 0.0025,
    #      0.0040, 0.0012, 0.0003, 0.0003, 0.0022, 0.0072, 0.0056, 0.0003, 0.0065,
    #      0.0060, 0.0008, 0.0003, 0.0005, 0.0003, 0.0024, 0.0005, 0.0003, 0.0003,
    #      0.0006, 0.0003, 0.0003, 0.0027, 0.0094, 0.0053, 0.0050, 0.0039, 0.0003,
    #      0.0040, 0.0067, 0.0013, 0.0065, 0.0004, 0.0030, 0.0003, 0.0014, 0.0003,
    #      0.0003, 0.0004, 0.0003, 0.0003, 0.0003, 0.0004, 0.0009, 0.0005, 0.0017,
    #      0.0013, 0.0003, 0.0009, 0.0011, 0.0008, 0.0003, 0.0003, 0.0018, 0.0003,
    #      0.0006, 0.0014, 0.0003, 0.0003, 0.0003, 0.0003, 0.0004, 0.0035, 0.0008,
    #      0.0003, 0.0003, 0.0003, 0.0005, 0.0026, 0.0003, 0.0003, 0.0013, 0.0010,
    #      0.0003, 0.0031, 0.0009, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0007,
    #      0.0003, 0.0003, 0.0003, 0.0041, 0.0003, 0.0003, 0.0017, 0.0016, 0.0003,
    #      0.0005, 0.0003, 0.0003, 0.0005, 0.0012, 0.0003, 0.0003, 0.0011, 0.0009,
    #      0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0006, 0.0003,
    #      0.0004, 0.0003, 0.0004, 0.0003, 0.0003, 0.0028, 0.0004, 0.0016, 0.0007,
    #      0.0026, 0.0003, 0.0003, 0.0003, 0.0011, 0.0006, 0.0003, 0.0003, 0.0003,
    #      0.0003, 0.0003, 0.0015, 0.0004, 0.0003, 0.0014, 0.0045, 0.0005, 0.0003,
    #      0.0003, 0.0076, 0.0019, 0.0003, 0.0003, 0.0015, 0.0037, 0.0020, 0.0003,
    #      0.0011, 0.0004, 0.0003, 0.0003, 0.0003, 0.0048, 0.0048, 0.0003, 0.0003,
    #      0.0040, 0.0003, 0.0029, 0.0082, 0.0017, 0.0026, 0.0008, 0.0008, 0.0026,
    #      0.0053, 0.0056, 0.0003, 0.0003, 0.0005, 0.0034, 0.0007, 0.0003, 0.0042,
    #      0.0069, 0.0004, 0.0003, 0.0087, 0.0003, 0.0003, 0.0049, 0.0097, 0.0003,
    #      0.0048, 0.0062, 0.0061, 0.0012, 0.0077, 0.0076, 0.0062, 0.0003, 0.0021,
    #      0.0003, 0.0085, 0.0003, 0.0003, 0.0057, 0.0003, 0.0053, 0.0019, 0.0007,
    #      0.0037, 0.0041, 0.0003, 0.0060, 0.0003, 0.0051, 0.0003, 0.0003, 0.0003,
    #      0.0006, 0.0053, 0.0006, 0.0051, 0.0003, 0.0085, 0.0003, 0.0049, 0.0075,
    #      0.0019, 0.0004, 0.0003, 0.0004, 0.0003, 0.0057, 0.0026, 0.0066, 0.0067,
    #      0.0036, 0.0040, 0.0016, 0.0008, 0.0003, 0.0019, 0.0073, 0.0003, 0.0011,
    #      0.0003, 0.0003, 0.0003, 0.0019, 0.0009, 0.0008, 0.0006, 0.0018, 0.0003,
    #      0.0074, 0.0006, 0.0008, 0.0003, 0.0026, 0.0003, 0.0004, 0.0028, 0.0003,
    #      0.0011, 0.0065, 0.0044, 0.0003, 0.0018, 0.0018, 0.0003, 0.0003, 0.0022,
    #      0.0061, 0.0003, 0.0003, 0.0003, 0.0032, 0.0003, 0.0003, 0.0003, 0.0003,
    #      0.0003, 0.0003, 0.0016, 0.0003, 0.0003, 0.0003, 0.0033, 0.0003, 0.0036,
    #      0.0003, 0.0022, 0.0013, 0.0003, 0.0055, 0.0031, 0.0003, 0.0102, 0.0005,
    #      0.0023, 0.0003, 0.0021, 0.0004, 0.0009, 0.0003, 0.0072, 0.0083, 0.0013,
    #      0.0005, 0.0003, 0.0090, 0.0079, 0.0003, 0.0003, 0.0054, 0.0083, 0.0008,
    #      0.0045, 0.0008, 0.0003, 0.0003, 0.0003, 0.0054, 0.0072, 0.0003, 0.0017,
    #      0.0046, 0.0016, 0.0009, 0.0007, 0.0006, 0.0031, 0.0070, 0.0067, 0.0005,
    #      0.0083, 0.0003, 0.0014, 0.0013, 0.0003, 0.0022, 0.0057, 0.0003, 0.0013,
    #      0.0003, 0.0013, 0.0012, 0.0016, 0.0003, 0.0009, 0.0003, 0.0003, 0.0070,
    #      0.0003, 0.0003, 0.0080, 0.0003, 0.0037, 0.0043, 0.0018, 0.0016, 0.0039,
    #      0.0005, 0.0003, 0.0010, 0.0006, 0.0025, 0.0020, 0.0003, 0.0003, 0.0073,
    #      0.0003, 0.0071, 0.0003, 0.0076, 0.0087, 0.0003, 0.0039, 0.0046, 0.0003,
    #      0.0081, 0.0003, 0.0063, 0.0003, 0.0041, 0.0037, 0.0055, 0.0011, 0.0004,
    #      0.0003, 0.0079, 0.0003, 0.0003, 0.0021, 0.0014, 0.0089, 0.0036, 0.0004,
    #      0.0003, 0.0003, 0.0004, 0.0007, 0.0040, 0.0004, 0.0003, 0.0070, 0.0003,
    #      0.0005, 0.0032, 0.0003, 0.0006, 0.0026, 0.0084, 0.0005, 0.0003, 0.0003,
    #      0.0003, 0.0013, 0.0011, 0.0004, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
    #      0.0007, 0.0068, 0.0015, 0.0003, 0.0004, 0.0053, 0.0005, 0.0015, 0.0003,
    #      0.0014, 0.0079, 0.0007, 0.0017, 0.0022, 0.0004, 0.0005, 0.0003, 0.0004,
    #      0.0004, 0.0003, 0.0009, 0.0070, 0.0004, 0.0003, 0.0003, 0.0003, 0.0078,
    #      0.0022]])
    readpath = '../dataset/'
    savepath = '../result/'
    weight = weight.squeeze(0).numpy()
    colon = heatmap(readpath, savepath)
    colon.visualizer(img_name, weight, 'Trans')

























