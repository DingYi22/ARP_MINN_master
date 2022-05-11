# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 20:49:38 2022

@author: Dingyi
"""

import cv2
import sys
import torch
import numpy as np
import random
import scipy.io as sio
import torchvision.transforms as transforms

from sklearn.model_selection import KFold

class DataGenerator(object):
    def __init__(self, dataset_name, seed, n_folds, 
                 isEnhance, isValid, train_ratio=0.8, shuffle=True):
        self.dataset_name = dataset_name
        self.seed = seed
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.isValid = isValid
        self.isEnhance = isEnhance
        self.train_ratio = train_ratio  if self.isValid  else 1

        # self.datasets = self.load_dataset()
        
    def load_dataset(self):
        if self.dataset_name.startswith('colon_cancer') is False and self.dataset_name.startswith('multi_colon') is False:
            raise Exception("this dataset can not be loaded!")
        print('Data is loading >>', end='')
        data = sio.loadmat('./dataset/'+self.dataset_name+'.mat')
        feas = data['x']['data'][0,0]
        labels = data['x']['label'][0,0]
        positions = data['x']['position'][0,0]
        nlabels = data['x']['nlabel'][0,0]
        names = data['x']['info_name'][0,0]
        
        bags = []
        for i in range(len(feas)):
            # bag = ()
            fea, label, position, nlabel, name = feas[i,0], labels[i][0], positions[i,0], nlabels[i,0], str(names[i,0][0])
            fea, label = fea.astype(np.float32), label.astype(np.float32).squeeze(0)
            position, nlabel = position.astype(np.float32), nlabel.astype(np.float32).squeeze(0)
            fea, label = torch.from_numpy(fea), torch.from_numpy(label)
            info = {'name':name, 'position': position, 'nlabel': nlabel}
            bag = (fea, label, info)
            bags.append(bag)
        print('>' * 5, end='')
        kf = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.seed)
        datasets = []
        for train_idx, test_idx in kf.split(bags):
            dataset = {}
            
            if self.isValid:                 
                num_train = int(self.train_ratio * len(train_idx))
                train_index, valid_index = train_idx[:num_train], train_idx[num_train:]
                dataset['train'] = [bags[ibag] for ibag in train_index]
                dataset['valid'] = [bags[ibag] for ibag in valid_index]
            else:
                dataset['train'] = [bags[ibag] for ibag in train_idx]
            print('>' * 3, end='')
            if self.isEnhance:            
                dataset['train'] = self._data_enhancer(dataset['train'])
            print('>' * 3, end='')
            dataset['test'] = [bags[ibag] for ibag in test_idx]
            datasets.append(dataset)
            print('>'*2, end='')
        print('>')
        return datasets
    
    
    def _data_enhancer(self, train_bags):
        transform_enhance = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation(degrees=90),
            transforms.ToTensor() ])
        for index, (fea, label, position) in enumerate(train_bags):
            for i in range(len(fea)):
                fea[i] = transform_enhance(fea[i])
            train_bags[index] = (fea, label, position)

        # for index, (fea, label, position) in enumerate(train_bags):
        #     imgs = fea.permute(0,2,3,1).numpy()
        #     for i in range(imgs.shape[0]):
        #         img = imgs[i, :, :, :]
        #         img = self._random_flip_img(img)
        #         img = self._random_rotate_img(img)
        #         imgs[i, :, :, :] = img
        #     fea_new = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        #     train_bags[index] = (fea_new, label, position)
        return train_bags

    def _random_flip_img(self, img, hori_chance=0.5, vert_chance=0.5):
        hori_prob, vert_prob = random.random(), random.random()
        
        if hori_prob < hori_chance and vert_prob < vert_chance:
            return img
        elif hori_prob >= hori_chance and vert_prob < vert_chance:
            flip_val = 0   # X axis
        elif hori_prob < hori_chance and vert_prob >= vert_chance:
            flip_val = 1   # Y axis
        else:
            flip_val = -1  # both axis
        
        if not isinstance(img, list):
            res = cv2.flip(img, flip_val) 
        else:
            res = []
            for img_item in img:
                img_flip = cv2.flip(img_item, flip_val)
                res.append(img_flip)
        return res
    
    def _random_rotate_img(self, img):
        rand_roat = np.random.randint(4)
        angle = 90*rand_roat
        center = (img.shape[0] / 2, img.shape[1] / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        img_en = cv2.warpAffine(img, rot_matrix, dsize=img.shape[:2], borderMode=cv2.BORDER_CONSTANT)
        return img_en
    
    def _random_crop(self, img, crop_size=(400, 400)):
        height, width = img.shape[:-1]
        dy, dx = crop_size
        X = np.copy(img)
        img_en = np.zeros(tuple([3, 400, 400]))
        if width < dx or height < dy:
            return None
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        img_en = X[y:(y + dy), x:(x + dx), :]
        return img_en


## test
if __name__ == '__main__': 
    
    ColonCancer = DataGenerator(dataset_name = 'colon_cancer', 
                                seed = 1, 
                                n_folds = 10, 
                                isEnhance = False, 
                                isValid = False, 
                                train_ratio=0.8, 
                                shuffle=True)
    
    datasets = ColonCancer.load_dataset()
    dataset = datasets[1]
    train_bags = np.array(dataset['train'][1][1])
    print(train_bags.shape)
    test_bags = dataset['test']
    # print(test_bags)

    
   
    
   
    
   
    

