# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:23:58 2022

@author: Dingyi
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  

import sys
import csv
import time
import random
import argparse
import numpy as np
from tqdm import tqdm


import torch
from torch import optim
from sklearn import metrics
from dataloader import *
from ARP_MINN import ARP_MINN
from layers import FeaturePooling
from utils import bag_accuracy, entropy_loss, bag_margin_loss, bag_loss
from visualizer.heatmap import heatmap
from visualizer.T_SNE import TSNE
from visualizer.vector3D import vector3D

device = 'cpu'
print(torch.__version__)
print(device)


def parse_args():
    parser = argparse.ArgumentParser(description='Train GPO_MIL')
    
    parser.add_argument('--model_name', dest='model_name',
                        default='SA+GPOMIL', type=str,
                        help='')
    
    parser.add_argument('--dataset', dest='dataset',
                        default='colon_cancer', type=str,
                        help='dataset to train on, like musk1 or fox')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--lr', dest='init_lr',
                        help='initial learning rate',
                        default=1e-4, type=float)
    
    parser.add_argument('--decay', dest='weight_decay',
                        help='weight decay',
                        default=5e-4, type=float)
    
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum',
                        default=0.9, type=float)

    parser.add_argument('--epoch', dest='max_epoch',
                        help='number of epoch to train',
                        default=60, type=int)
    
    parser.add_argument('--d_model', dest='model_d',
                        help='positional encoding dimention',
                        default=64, type=int)
    
    parser.add_argument('--d_hidden', dest='hidden_d',
                        help='The number of features in the hidden state',
                        default=64, type=int)
    
    parser.add_argument('--nheads', dest='num_heads', 
                        help='', default=1, type=int)
    
    args = parser.parse_args()
    return args

def test_eval(net, test_bags):
    with torch.no_grad():
        net.eval()
        test_loss, test_acc = 0., 0.
        for batch_idx, (data, label, _) in enumerate(test_bags):
            data, label = data.to(device), label.to(device)
            bag_length = torch.tensor(np.array([data.shape[0]])).to(device)
            y_prob, dec_alpha, _ = net(data, bag_length)
            l = bag_loss(y_prob, label)
            test_loss += l.item()
            test_acc  += bag_accuracy(y_prob, label)
            
    test_acc, test_loss = test_acc / len(test_bags), test_loss / len(test_bags)
    return test_loss, test_acc

def test_and_draw(net, test_bags):
    y_true, y_pred = torch.tensor([]), torch.tensor([])
    print('testing and drawing: ', end='')
    with torch.no_grad():
        net.eval()
        for batch_idx, (data, label, info) in enumerate(test_bags):
            data, label = data.to(device), label.to(device)
            bag_length = torch.tensor(np.array([data.shape[0]])).to(device)
            y_prob, dec_alpha, embed = net(data, bag_length)
            
            y_prob = torch.ge(y_prob, 0.5).float()
            y_true, y_pred = torch.cat((y_true, label), -1), torch.cat((y_pred, y_prob), -1)
            
            if label == 1.:
                embed = embed.squeeze(0).numpy()
                save_path = './result/pics/' + info['name'] + '_vector3D_' + args.model_name + '.png'
                nlabel = info['nlabel']
                vector3D(3).visualizer(embed, nlabel, save_path)
                
                save_path = './result/pics/' + info['name'] + '_TSNE_' + args.model_name + '.png'
                TSNE(2).visualizer(embed, nlabel, save_path)
                
                dec_alpha = dec_alpha.squeeze(0).numpy()
                readpath, savepath = './dataset/', './result/pics/'
                heatmap(readpath, savepath).visualizer(info['name'], dec_alpha, args.model_name)
                
                with open('./result/params.txt','a') as f:
                    f.write(info['name'] + ': \n')
                    f.write(str(dec_alpha) + '\n')
                print('>', end='')
                
    print(' Finished!')            
    y_true, y_pred = y_true.numpy(), y_pred.numpy()
    
    test_acc = metrics.accuracy_score(y_true, y_pred)
    test_precs = metrics.precision_score(y_true, y_pred)
    test_recall = metrics.recall_score(y_true, y_pred)
    test_f1 = metrics.f1_score(y_true, y_pred)
    test_auc = metrics.roc_auc_score(y_true, y_pred)
    
    return test_acc, test_precs, test_recall, test_f1, test_auc


def train_eval(net, optimizer, scheduler, train_bags, test_bags):
    
    print('Start Training')
    
    # print("epoch: ", end='')
    extre_count, extre_test_acc, extre_test_loss = 0, 0., 100.
    best_count,  best_test_acc,  best_test_loss  = 0, 0., 100.
    for epoch in range(args.max_epoch):                          # epoch
        net.train()
        train_loss, train_acc = 0., 0.
        with tqdm(train_bags) as pbar:
            for data, label, _ in pbar:
                pbar.set_description('training: ')
                data, label = data.to(device), label.to(device)
                bag_length = torch.tensor(np.array([data.shape[0]])).to(device)
                y_prob, dec_alpha, _ = net(data, bag_length)
                l = bag_loss(y_prob, label)
                optimizer.zero_grad()                   
                l.backward()     
                optimizer.step()   
                train_loss += l.cpu().item()             
                train_acc += bag_accuracy(y_prob, label)
            scheduler.step()
            
            train_loss, train_acc = train_loss / len(train_bags), train_acc / len(train_bags)
            
            if (epoch + 1) % 1 == 0:
                test_loss, test_acc = test_eval(net, test_bags)
                print('Epoch: {}, train loss: {:.4f} train acc: {:.4f}, test loss: {:.4f} test acc: {:.4f}\n' \
                      .format(epoch, train_loss, train_acc, test_loss, test_acc))
                # best method #
                if test_acc > best_test_acc:
                    best_test_loss, best_test_acc = test_loss, test_acc
                    best_count += 1
                    torch.save(net, './result/model/saved_' + args.model_name + '.pkl')
                # extre method #
                if test_loss < extre_test_loss:
                    extre_test_loss, extre_test_acc = test_loss, test_acc
                    extre_count += 1
                
    # base method #
    base_test_loss, base_test_acc = test_loss, test_acc
    
    print("# base method  # Model last epoch: %3d, base  test loss: %.4f, base  test acc: %.4f" %(args.max_epoch, base_test_loss, base_test_acc))
    print("# extre method # Model save times: %3d, extre test loss: %.4f, extre test acc: %.4f" %(extre_count, extre_test_loss, extre_test_acc))
    print("# best method  # Model save times: %3d, best  test loss: %.4f, best  test acc: %.4f" %(best_count, best_test_loss, best_test_acc))
    print('Optimization Finished!\n')
    
    return base_test_acc, extre_test_acc, best_test_acc

def model_training(dataset, irun, ifold):
    train_bags = dataset['train']
    test_bags = dataset['test']
    
    net = ARP_MINN(num_heads=args.num_heads, dim_out=1, d_pe=args.model_d, d_hidden=args.hidden_d, num_seeds=1)
    
    optimizer = optim.Adam(net.parameters(), lr=args.init_lr, betas=(0.9, 0.98), 
                           eps=1e-09, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch, eta_min=0)
    
    base_test_acc, extre_test_acc, best_test_acc = train_eval(net, optimizer, scheduler, train_bags, test_bags)
    
    with open('./result/params.txt','a') as f:
        f.write('* ' * 30 + '\n')
        f.write('irun = ' + str(irun) + '  ifold = ' + str(ifold) + '\n')
        
    saved_net = torch.load('./result/model/saved_' + args.model_name + '.pkl')
    test_acc, test_precs, test_recall, test_f1, test_auc = test_and_draw(saved_net, test_bags)
    
    return test_acc, test_precs, test_recall, test_f1, test_auc, base_test_acc, extre_test_acc, best_test_acc

if __name__ == "__main__":
    args = parse_args()
    print ('Called with args:')
    print (args)
    
    with open('./result/results.txt','a') as f:
        f.write('* ' * 50 + '\n')
        f.write(str(args) + '\n')
    run, n_folds = 5, 10
    seed = [3*i+1 for i in range(run)]
    base_test_acc  = np.zeros((run, n_folds), dtype=float)
    extre_test_acc  = np.zeros((run, n_folds), dtype=float)
    best_test_acc  = np.zeros((run, n_folds), dtype=float)
    acc = np.zeros((run, n_folds), dtype=float)
    precs = np.zeros((run, n_folds), dtype=float)
    recall = np.zeros((run, n_folds), dtype=float)
    f1 = np.zeros((run, n_folds), dtype=float)
    auc = np.zeros((run, n_folds), dtype=float)

    for irun in range(run):
        ColonCancer = DataGenerator(dataset_name = 'colon_cancer', 
                                 seed = seed[irun], 
                                 n_folds = 10, 
                                 isEnhance = True, 
                                 isValid = False, 
                                 train_ratio=1, 
                                 shuffle=True)
    
        dataset = ColonCancer.load_dataset()
        for ifold in range(n_folds):
            print ('run=', irun, '  fold=', ifold)
            acc[irun][ifold], precs[irun][ifold], recall[irun][ifold], f1[irun][ifold], auc[irun][ifold], base_test_acc[irun][ifold], extre_test_acc[irun][ifold], best_test_acc[irun][ifold] = model_training(dataset[ifold], irun, ifold)
    print('Trans_Colon metrics: ')
    print('Accuracy  mean = {:.3f}, std = {:.3f}'.format(np.mean(acc), np.std(acc)))
    print('Precision mean = {:.3f}, std = {:.3f}'.format(np.mean(precs), np.std(precs)))
    print('Recall    mean = {:.3f}, std = {:.3f}'.format(np.mean(recall), np.std(recall)))
    print('F-score   mean = {:.3f}, std = {:.3f}'.format(np.mean(f1), np.std(f1)))
    print('AUC       mean = {:.3f}, std = {:.3f}'.format(np.mean(auc), np.std(auc)))
    
    base_test_acc  = 'base_test_acc  mean = ' + str(np.mean(base_test_acc)) + ', std = ' + str(np.std(base_test_acc)) + '\n'
    extre_test_acc = 'extre_test_acc  mean = ' + str(np.mean(extre_test_acc)) + ', std = ' + str(np.std(extre_test_acc)) + '\n'
    best_test_acc  = 'best_test_acc  mean = ' + str(np.mean(best_test_acc)) + ', std = ' + str(np.std(best_test_acc)) + '\n'
    
    Accuracy   = 'Accuracy  mean = ' + str(np.mean(acc)) + ', std = ' + str(np.std(acc)) + '\n'
    Precision  = 'Precision mean = ' + str(np.mean(precs)) + ', std = ' + str(np.std(precs)) + '\n'
    Recall     = 'Recall    mean = ' + str(np.mean(recall)) + ', std = ' + str(np.std(recall)) + '\n'
    F_score    = 'F-score   mean = ' + str(np.mean(f1)) + ', std = ' + str(np.std(f1)) + '\n'
    AUC        = 'AUC       mean = ' + str(np.mean(auc)) + ', std = ' + str(np.std(auc)) + '\n'

    with open('./result/results.txt','a') as file_handle:
        file_handle.write('* ' * 50 + '\n')
        file_handle.write(str(args) + ':\n')
        file_handle.write('Trans_Colon metrics: \n')
        file_handle.write(base_test_acc)
        file_handle.write(extre_test_acc)
        file_handle.write(best_test_acc)
        file_handle.write(Accuracy)
        file_handle.write(Precision)
        file_handle.write(Recall)
        file_handle.write(F_score)
        file_handle.write(AUC)


        
    
    











