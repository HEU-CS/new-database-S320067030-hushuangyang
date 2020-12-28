#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *

# ArgumentParser 创建解析器
parser = argparse.ArgumentParser()
# add_argument 添加参数
# 数据集
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
# 输入分支数
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
# 隐藏状态大小
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
# 训练的次数
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
# 学习率
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
# 学习率的下降率
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
# 学习率下降时的步数
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
# 惩罚值
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
# GNN传播步长
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')

parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
# 只使用全局偏好去预测
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
# 验证
parser.add_argument('--validation', action='store_true', help='validation')
# 划分训练集的一部分做验证集
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
# 解析添加的参数
opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True)
    # <class 'tuple'>: ([[282], [281, 308], [281], [58, 58, 58, 230, 230, 230, 246, 230], [58, 58, 58, 230, 230, 230, 246], [58, 58, 58, 230, 230, 230], [58, 58, 58, 230, 230],
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    '''
        SessionGraph(
      (embedding): Embedding(310, 100)
      (gnn): GNN(
        (linear_edge_in): Linear(in_features=100, out_features=100, bias=True)
        (linear_edge_out): Linear(in_features=100, out_features=100, bias=True)
        (linear_edge_f): Linear(in_features=100, out_features=100, bias=True)
      )
      (linear_one): Linear(in_features=100, out_features=100, bias=True)
      (linear_two): Linear(in_features=100, out_features=100, bias=True)
      (linear_three): Linear(in_features=100, out_features=1, bias=False)
      (linear_transform): Linear(in_features=200, out_features=100, bias=True)
      (loss_function): CrossEntropyLoss()
    )
    '''
    model = trans_to_cuda(SessionGraph(opt, n_node))  # opt....   n_node 310会话中点的数量，也就是涉及项的数量

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
