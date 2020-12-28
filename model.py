#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
# Pytorch中神经网络模块化接口nn,nn.Module是nn中十分重要的类,包含网络各层的定义及forward方法。
from torch import nn

"""
定义自已的网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中，
    不具有可学习参数的层(如ReLU)可放在构造函数中，也可不放在构造函数中(而在forward中使用nn.functional来代替)
    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
    在forward函数中可以使用任何Variable支持的函数，毕竟在整个pytorch构建的图中，是Variable在流动。还可以使用
    if,for,print,log等python语法.
    
    注：Pytorch基于nn.Module构建的模型中，只支持mini-batch的Variable输入方式，
    比如，只有一张输入图片，也需要变成 N x C x H x W 的形式：
    
    input_image = torch.FloatTensor(1, 28, 28)
    input_image = Variable(input_image)
    input_image = input_image.unsqueeze(0)   # 1 x 1 x 28 x 28
    
"""
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        # 第一种方法__init__()方法是一种特殊的方法，被称为类的构造函数或初始化方法，当创建了这个类的实例时就会调用该方法
        # self 代表类的实例，self 在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数。
        # super() 函数是用于调用父类(超类)的一个方法    
        super(GNN, self).__init__()
        # step开始是1 GNN向前传播的步数
        self.step = step
        self.hidden_size = hidden_size  # 100
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        # 将参数变为可训练的
        # torch.tensor是一个包含多个同类数据类型数据的多维矩阵。
        # 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面，所以经过类型转换这个self.XXX变成了模型的一部分，成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))
        # 有关nn.Linear的解释：torch.nn.Linear(in_features, out_features, bias=True)，对输入数据做线性变换：y=Ax+b
        # 形状：输入: (N,in_features)  输出： (N,out_features)
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    # 输入A-->(batch_size , max_n_node , 2*max_n_node):节点间的对应转移矩阵，包括出边和入边的对应的邻接矩阵
    # hidden-->（batch_size,max_n_node,embeding_size）:每个节点的嵌入向量表示
    # 输出：hy-->(batch_size,max_n_node,embeding_size=hidden_size):一次更新后的节点的向量表示
    def GNNCell(self, A, hidden):
        # 一次GNN的迭代
        # A-->实际上是该批数据图矩阵的列表A-->shape(batch_size x max_n_node x 2*max_n_node)
        # hidden--> eg(batch_size,max_n_node,embeding_size=hidden_size)
        # 后面所有的5?代表这个维的长度是该批唯一最大类别长度(类别数目不足该长度的会话补零)，根据不同批会变化
        # 有关matmul的解释：矩阵相乘，多维会广播相乘
        # 既然Asi代表输入与输出列;则将As分成输入与输出两部分一次同时计算
        '''  A:torch.Size([100, 6, 12])
          tensor([[[0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 1.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 1.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000]],

                [[0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.5000, 0.5000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000]],

                [[0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 1.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 1.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000]],

                [[0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 1.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 1.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000]]])
        '''
        # torch.matmul是tensor的乘法，输入可以是高维的。
        # AS,i表示表示的是节点Vs,i分别在Ain、Aout对应的两列。
        # torch.Size([100, 7, 100]) A[:, :, :A.shape[1]]相当于A(0,0,列长度)  A.shape[0]:行长度  A.shape[1]:列长度
        # A:100*6*12 hidden:100*6*100    input:100*6*100 output:100*6*100  inputs:100*6*200
        # [batch,height,width] * [batch,width,height] = [batch,height,height]
        # A.shape[1]:6    A.shape[2]:12  A.shape[1]: 2 * A.shape[1]]:7-12 self.linear_edge_in(hidden):100 *6*100
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        # as,i是Ain和Aout拼接而成的
        inputs = torch.cat([input_in, input_out], 2)  # torch.Size([100, 7, 200])
        # 乘以权重w_ih加上误差b_ih  w_ih:300*200
        gi = F.linear(inputs, self.w_ih, self.b_ih)  # torch.Size([100, 7, 300]) )(包含了W_z*a_(s.i)^t和W_r*a_(s,i)^t的操作
        gh = F.linear(hidden, self.w_hh, self.b_hh)  # torch.Size([100, 7, 300])  （U_z*v_i^(t-1)和U_r*v_i^(t-1)
        # torch.chunk(tensor, chunk_num, dim)将tensor按dim（行或列）分割成chunk_num个tensor块，返回的是一个元组。
        # i_r : Wz * As,i    h_r : Uz * vi    i_i : Wz * As,i  h_i : Uz * vi(t-1)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)  # torch.Size([100, 6, 100])
        # 100*6*100
        inputgate = torch.sigmoid(
            i_i + h_i)  # 更新门:控制当前状态需要从历史状态中保留多少信息（不经过非线性变换），以及需F要从候选状态中接受多少新信息。inputgate-->(100,6,100)  原文公式(2)
        resetgate = torch.sigmoid(i_r + h_r)  # 重置门:控制候选状态h_t是否依赖于上一时刻的状态h_t-1.resetgate-->(100,6,100)      原文公式(3)
        newgate = torch.tanh(
            i_n + resetgate * h_n)  # newgate-->(100,5?,100)  原文公式(4)  i_n:Wo * As,i   h_n:Uo * resetgate
        hy = newgate + inputgate * (hidden - newgate)  # hy-->(100,5?,100)    原文公式(5)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)  # torch.Size([100, 7, 100])
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        # opt Namespace(batchSize=100, dataset='sample', epoch=30, hiddenSize=100, l2=1e-05, lr=0.001, lr_dc=0.1, lr_dc_step=3, nonhybrid=False, patience=10, step=1, valid_portion=0.1, validation=False)
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node  # 310
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node,
                                      self.hidden_size)  # 这是一个矩阵类，里面初始化了一个随机矩阵，矩阵的长是字典的大小，宽是用来表示字典中每个元素的属性向量，向量的维度根据你想要表示的元素的复杂度而定。长:n_node；宽:hidden_size
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()  # 损失函数定义为交叉熵函数   pytorch里交叉熵已经包含了softmax操作了，不需要自己另外加
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        # 从每一批（会话）里面选最后一个项目，即Vn也就是局部偏好。torch.sum(mask, 1)-->(batch_size,)减一指对每一个元素都减一，
        # 因为下标左闭右开，所以减一。 这是最后一个动作对应的位置，即文章中说的局部偏好
        # mask-->(batch_size,max session length,)标记有动作的位置   torch.sum(mask, 1)对一行mask求和
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # 生成一组0 - mask行长度的序列
        q1 = self.linear_one(ht).view(ht.shape[0], 1,
                                      ht.shape[1])  # 100*1*100 重塑张量的形状ht.shape[0]组 1*ht.shape[1]的张量 W*V_i
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size 100*16*100    W*V_n
        alpha = self.linear_three(
            torch.sigmoid(q1 + q2))  # 加法向维度高的一个对齐，维度数据不够的一方则，从不够的维度开始数据到对齐。所以呢这个q1+q2是将每个会话的每个节点与自己所在会话的最后一个节点相加
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)  # *号是元素对应相乘，(100,100) 即全局偏好 原文中公式(6)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))  # 原文中公式(7) #torch.cat:拼接
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))  # 原文中公式(8) 注意力机制中的加性模型
        return scores

    # 输入inputs-->(batch_size,max_n_node)单个点击动作序列的唯一类别并按照批最大唯一类别长度补全0列表
    # A-->(batch_size , max_n_node , 2*max_n_node):节点间的对应转移矩阵，包括出边和入边的对应的邻接矩阵
    # 输出hidden-->(batch_size,max_n_node,embeding_size):得到的最终的节点的嵌入式表示
    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)  # 随机初始化每个节点的嵌入向量
        return hidden


# GPU状态查询
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    '''alias_inputs tensor([[1, 0, 0,  ..., 0, 0, 0],
        [1, 0, 0,  ..., 0, 0, 0],
        [1, 0, 0,  ..., 0, 0, 0],
        ...,
        [1, 0, 0,  ..., 0, 0, 0],
        [1, 1, 0,  ..., 0, 0, 0],
        [1, 0, 0,  ..., 0, 0, 0]])

        items: tensor([[  0,  85,   0,   0,   0],
        [  0,  99,   0,   0,   0],
        [  0,   7,   0,   0,   0],
        [  0, 231,   0,   0,   0],
        [  0,  21,   0,   0,   0],
        [  0,  42,   0,   0,   0],
        [  0, 135, 136,   0,   0],
        [  0, 148, 250,   0,   0],
        [  0, 122, 130, 240,   0],
        [  0, 150, 151,   0,   0],
    '''
    # 输出的items是以唯一动作序列列表为元素的列表;alias_inputs 有i->j的动作转移则为1-->(batchSize , size of u_input)返回的动作集合的对应的位置下标们;A-->(batch_size , max_n_node , 2*max_n_node):节点间的对应转移矩阵，包括出边和入边的对应的邻接矩阵
    # alias_inputs-->(batch_size,length of session)返回动作集合对应的动作标签的位置下标
    # A-->(batch_size , max_n_node , 2*max_n_node):节点间的对应转移矩阵，包括出边和入边的对应的邻接矩阵
    # items-->(batch_size,max_n_node)单个点击动作序列的唯一类别并按照批最大类别补全0;输出的items是以唯一动作序列列表为元素的列表
    # mask-->(batch_size,max session length,)标记有动作的位置 有1无0
    # targets-->(batch_size,)索引对应的目标数据列表
    alias_inputs, A, items, mask, targets = data.get_slice(i)  # 返回动作序列对应唯一动作集合，的位置下标
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    '''A torch.Size([100, 8, 16]) 100组数据，每组8个结点， 16=8入边8出边
        <class 'list'>:
           array(
           [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
            array(
           [[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,0. ],
           [0. , 0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0.5, 0.5, 0. , 0. , 0. ,0. ],
           [0. , 0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0.5, 0.5, 0. , 0. , 0. ,0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,0. ],
        '''
    A = trans_to_cuda(torch.Tensor(A).float())  #
    mask = trans_to_cuda(torch.Tensor(mask).long())
    '''  items是以唯一动作序列列表为元素的列表
    item tensor([[  0,  29,   0,   0,   0,   0,   0],
        [  0, 108, 118,   0,   0,   0,   0],
        [  0, 222, 223, 224, 225, 226,   0],
        [  0,  12,  44,  67, 140, 157,   0],
        [  0, 210,   0,   0,   0,   0,   0],
        [  0, 253, 254,   0,   0,   0,   0],
        [  0, 264,   0,   0,   0,   0,   0],
        [  0, 281,   0,   0,   0,   0,   0],
        [  0, 172, 198,   0,   0,   0,   0],
        [  0, 202,   0,   0,   0,   0,   0],
        [  0, 231,   0,   0,   0,   0,   0],
        [  0, 247, 291,   0,   0,   0,   0],
        [  0, 209,   0,   0,   0,   0,   0],
        [  0, 137,   0,   0,   0,   0,   0],
        [  0, 197, 249, 250,   0,   0,   0],
        [  0, 227, 228,   0,   0,   0,   0],
        [  0, 102,   0,   0,   0,   0,   0],
        [  0,  73,  74,  75,   0,   0,   0],
        [  0, 150, 151, 152,   0,   0,   0],
        [  0, 107, 108,   0,   0,   0,   0],
        [  0, 139,   0,   0,   0,   0,   0],
        [  0,  22,   0,   0,   0,   0,   0],
        [  0, 130,   0,   0,   0,   0,   0],
        [  0, 113,   0,   0,   0,   0,   0],
        [  0,  58,   0,   0,   0,   0,   0],
    '''
    # 这里调用了SessionGraph的forward函数,返回维度数目(100,5?,100),这个model是SessionGraph类型的,(batch_size,max_n_node,embeding_size=hidden_size),即每一批中的每一个节点，都得到了一个嵌入式向量表示
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]  # 选择第这一批第i个样本对应类别序列的函数,返回(max_n_node,hiddenSize)
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


"""
训练和测试函数将根据给定的模型和训练以及测试数据，执行训练和测试步骤。
输入：
model-->(SessionGraph):传入的模型SessionGraph
train_data-->(Data):
test_data-->(data):训练数据和测试数据
epoch-->(num):第几个epoch
输出：
hit-->(num):正确标签是否在预测的前二十个的个数统计P@20。（均是在测试集合中得出）
mrr-->(num):MRR@20，正确推荐项目的倒数排名的平均数。当等级超过20时，倒数排名设置为0
total_loss:一次epoch的误差总和
"""


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        # 设置梯度为0  前一步的损失清零
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)  # 公式（8）、（9）
        loss.backward()  # 反向传播
        model.optimizer.step()  # 优化
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []  # P@20 MRR@20
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):  # test_data.mask有序列的位置是[1],没有动作序列的位置是[0]
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:  # numpy下标从0开始,返回比较数组的下标(x,y)
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100  # np.mean()函数会将True作为1，False做为0
    mrr = np.mean(mrr) * 100
    return hit, mrr
