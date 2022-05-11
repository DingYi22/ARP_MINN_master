# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 20:49:38 2022

@author: Dingyi
"""


import torch
import numpy as np
import sys
import random
import scipy.io as sio
from sklearn.model_selection import KFold
device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset(dataset_nm, n_folds, seed=1):
    # load data from file
    if dataset_nm.startswith('musk1norm_matlab') is True or dataset_nm.startswith('trec_1_200x200_matlab') is True:
        raise Exception("this function can not load musk1norm_matlab !")
        
    data = sio.loadmat('./dataset/'+dataset_nm+'.mat')
    
    ins_fea = data['x']['data'][0,0]     # feature array [6598, 166]

    if dataset_nm.startswith('musk'):
        bags_nm = data['x']['ident'][0,0]['milbag'][0,0]
    else:
        bags_nm = data['x']['ident'][0,0]['milbag'][0,0][:,0]   

    ins_label = data['x']['nlab'][0,0][:,0] - 1   # label array [6598, ]
    ins_label = ins_label.astype(np.float32)


    if dataset_nm.startswith('newsgroups') is False:
        mean_fea = np.mean(ins_fea, axis=0, keepdims=True)+1e-6
        std_fea = np.std(ins_fea, axis=0, keepdims=True)+1e-6
        ins_fea = np.divide(ins_fea-mean_fea, std_fea)
        ins_fea = ins_fea.astype(np.float32)


    ins_idx_of_input = {}            # store instance index of input 示例索引
    for id, bag_nm in enumerate(bags_nm):
        if bag_nm in ins_idx_of_input: ins_idx_of_input[bag_nm].append(id)
        else:                                ins_idx_of_input[bag_nm] = [id]

    ins_fea_tensor = torch.from_numpy(ins_fea)
    ins_label_tensor = torch.from_numpy(ins_label)
    

    bags_fea = []
    for bag_nm, ins_idxs in ins_idx_of_input.items():
        bag_fea = ()
        bag_fea = (ins_fea_tensor[ins_idxs], ins_label_tensor[ins_idxs])
        bags_fea.append(bag_fea)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    datasets = []
    for train_idx, test_idx in kf.split(bags_fea):
        dataset = {}
        dataset['train'] = [bags_fea[ibag] for ibag in train_idx]
        dataset['test'] = [bags_fea[ibag] for ibag in test_idx]
        datasets.append(dataset)
    
    return datasets



if __name__ == '__main__': 
    
    datalist = ('musk1', 'musk2',
                'elephant', 'fox', 'tiger','ucsb_breast', 'messidor',
                'newsgroups164', 'newsgroups165', 'newsgroups166', 'newsgroups167',
                'newsgroups168', 'newsgroups169', 'newsgroups170', 'newsgroups171', 
                'newsgroups172', 'newsgroups173', 'newsgroups174', 'newsgroups175', 
                'newsgroups176', 'newsgroups177', 'newsgroups178', 'newsgroups179', 
                'newsgroups180', 'newsgroups181', 'newsgroups182', 'newsgroups183', )
    
    for i in range(0, 25):
        print("dataset: ", str(datalist[i]))
        dataset_nm, n_folds = datalist[i], 10
        datasets = load_dataset(dataset_nm, n_folds)
            
        batch_size = 1

        if sys.platform.startswith('win'): num_workers = 0
        else:                              num_workers = 4
            
        train_iter = torch.utils.data.DataLoader(datasets[0]['train'], batch_size=batch_size, 
                                                 shuffle=True, num_workers=num_workers)
        test_iter = torch.utils.data.DataLoader(datasets[0]['test'], batch_size=batch_size, 
                                                shuffle=False, num_workers=num_workers)
           
        for X, y in train_iter:
            print(X, y)
            print(X.shape, y.shape)
            print("\n")
            break