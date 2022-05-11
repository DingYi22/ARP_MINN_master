# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:23:09 2022

@author: Dingyi
"""

import sys
import math
import numpy as np
from IPython import display
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
from torch import nn


def bag_accuracy(y_pred, y_true):
    y_true = torch.mean(y_true, axis=0)
    y_pred = torch.mean(y_pred, axis=0)
    acc = torch.equal(y_true, torch.round(y_pred))+0
    return acc

def bag_loss(y_pred, y_true):
    y_true = torch.mean(y_true, axis=0)
    y_pred = torch.mean(y_pred, axis=0)
    loss = torch.mean(nn.BCELoss(reduction='sum')(y_pred, y_true), axis=-1)
    return loss

def bag_margin_loss(y_pred, y_true):
    y_true = torch.mean(y_true, axis=0)
    y_pred = torch.mean(y_pred, axis=0)
    loss = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2   
    return loss

class entropy_loss(nn.Module):
    def __init__(self):
        super(entropy_loss, self).__init__()
        
    def forward(self, y_pred, y_true):
        return bag_loss(y_pred, y_true)
    

def AttentionVisualize(dataset_name, irun, ifold, batch_idx, weight, label, y_pred):
    vmin, vmax = 0., 13  
    # weight = unitization(weight)
    weight = weight.numpy()
    weight_min , weight_max  = np.min(weight), np.max(weight)
    # weight_min , weight_max  = 0.075, 0.17  # ~~1/13, 1/6
    weight = ( (vmax - vmin) * (weight - weight_min) / (weight_max - weight_min) )
    
    plt.figure(dpi=300) 
    
    # cm_list = mpl.colors.ListedColormap(['#FFFFFF', '#E1FFFF', '#ADD8E6', '#B0E0E6', '#00BFFF', '#4682B4', '#1E90FF'])
    cm_list = mpl.colors.ListedColormap(['#F7FBFF', '#F0F8FF', '#ECFBFF', '#DFEFFF', '#E0ECFF', '#D0E8FF', 
                                          '#BFDFFF', '#B9D5FF', '#B0D8FF', '#9DCEFF', '#8CC6FF', '#80BFFF', 
                                          '#6CB6FF', '#5BADFF', '#559FFF', '#48A4FF', '#399CFF', '#2F7FBC'])  #    vmin, vmax = 0., 13 
    
    # cm_list = mpl.colors.ListedColormap(['#470C61', '#481D70', '#4A2A79', '#414688', '#21858C', '#1D9F86', 
    #                                      '#38B874', '#90D643', '#A5DB33', '#FDE523'])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    pc_kwargs = {'rasterized': False, 'cmap': cm_list, 'norm': norm}  # 'cmap': 'gray' 'viridis'
    im = plt.pcolormesh(weight, **pc_kwargs)
    plt.colorbar(im, shrink=0.6)
    
    row, col = weight.shape
    plt.axis([0, row, 0, col])
    
    
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()                  
    
    plt.axis('scaled') 
    
    name = 'enc_weight ' if weight.shape[0] == weight.shape[1] else 'dec_weight_'
    
    plt.savefig('result/'+dataset_name+"/" + name + str(irun) + '-' + str(ifold) + '-' + str(batch_idx) +'(' + str(label) + ',' + str(y_pred) + ')' +'.png', bbox_inches='tight', dpi=300)
    # plt.show()