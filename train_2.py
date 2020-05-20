# -*- coding: utf-8 -*-
# @Time           : 2018/4/10 17:07
# @Author         : Barry
# @Email          : wcf.barry@foxmail.com
# @Modify Time    : 2020/5/20 16:38
# @Mender         ：simon
# @Email          : simongeng1@163.com
# @os             : ubuntu
# @File           : train_2.py
# @Software       : PyCharm

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import data
import model2 as model
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

load = False  #是否家在已训练过的模型
n_epoch =100 #迭代次数
isPlot = True  #是否画图并保存
df = pd.DataFrame(columns=['耗时', '训练集正确率', '测试集正确率'])   #记录最终正确率和耗时
criterion = nn.CrossEntropyLoss()

dr = "first_4_dr_02_0.05."  #所有保存文件和加载文件的前缀
#主循环中的超参数组，方便一次得迭代多组超参数，第一个参数是学习率，第二个参数是正则化系数
T = [[0.0001,0.005],[0.0002, 0.01],[0.0003, 0.005],[0.00005, 0.0075],[0.001, 0.001],[0.00075, 0.002]]
#T = [[0.0002,0]]
lrgamma =0.94 # 学习率衰减
batch_size = 200 #batch大小，奈何修改者显卡较差（4g显存），无法支持较大的batch
load = True   #是否加载已训练过的模型
accumlation_step = 4  #梯度累加次数 ，原理可以参考博客 https://www.cnblogs.com/qiulinzhang/p/11169236.html

def train(epoch):
    net.train()  # 网络处于训练模式，会导致dropout启用
    correct = 0
    sum = 0
    for batch_index, (datas, labels) in enumerate(trainloader, 0):
        labels = labels.max(1)[1]
        datas = Variable(datas).float()
        datas = datas.view(-1, 1, 128, 128)
        labels = Variable(labels).long()
        if torch.cuda.is_available():
            datas = datas.cuda()
            labels = labels.cuda()

        outputs = net(datas)
        loss = criterion(outputs, labels)
        if (batch_index + 1) == (len(trainloader) ) :   #记录loss
        # if (batch_index+1) % (len(trainloader) // 4) == 0:
            loss_.append(loss.item())
        loss = loss / accumlation_step
        loss.backward()
        if (batch_index + 1) % accumlation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        pred_choice = outputs.data.max(1)[1]
        correct += pred_choice.eq(labels.data).cpu().sum()
        sum += len(labels)
        if batch_index % 50 == 0 or batch_index == (len(trainloader) -1):
            b = int(correct) / int(sum)
            print('batch_index: [%d/%d]' % (batch_index, len(trainloader)),
                  'Eval epoch: [%d]' % epoch,
                  'correct/sum:%d/%d, %.4f' % (correct, sum, b))
        """
        if (batch_index+1) % (len(trainloader)//4) == 0:
            acc.append(b)
        """
    return b    #返回正确率

def eval(epoch):
    net.eval()  # 弯网络处于测试模式，dropout停用，BN放射变换停止
    correct = 0
    sum = 0
    for batch_index, (datas, labels) in enumerate(evalloader, 0):
        labels = labels.max(1)[1]
        datas = Variable(datas).cuda().float()
        datas = datas.view(-1, 1, 128, 128)
        labels = Variable(labels).cuda().long()
        # optimizer.zero_grad()
        outputs = net(datas)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        pred_choice = outputs.data.max(1)[1]
        correct += pred_choice.eq(labels.data).cpu().sum()
        sum += len(labels)
        if batch_index % 5 == 0 or batch_index == (len(evalloader) - 1):
            a = int(correct) / int(sum)
            print('batch_index: [%d/%d]' % (batch_index, len(evalloader)),
                  'Eval epoch: [%d]' % epoch,
                  'correct/sum:%d/%d, %.4f' % (correct, sum,a))
        """        
        if (batch_index+1) % (len(evalloader)//4) == 0:
            acc2.append(a)
        """
    return a     #返回正确率


if __name__ == '__main__':

    for t in T:
        print(t)
        print("***********************************************************************")
        acc = []    #测试集正确率记录
        loss_ = []  #loss记录
        acc2 = []
        net = model.net()
        if torch.cuda.is_available():
            net.cuda()
        trainset = data.TrainSet(eval=False)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        evalset = data.TrainSet(eval=True)
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=True)
        starttime = time.time()

        if load:
            checkpoint = model.load_checkpoint(t2 = t,dr2 = dr)
            net.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            acc = checkpoint["acc"]
            acc2= checkpoint["acc2"]
            loss_ = checkpoint["loss_"]
        else:
            start_epoch = 0

        # 设置优化器
        optimizer = optim.Adam(net.parameters(), lr=t[0], betas=(0.9, 0.999), weight_decay=t[1])
        # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=1e-1, weight_decay=1e-4)
        ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lrgamma)    #学习率衰减 指数
        for epoch in range(start_epoch, n_epoch):
            b = train(epoch)
            acc.append(b)

            a = eval(epoch)
            acc2.append(a)
            # 保存参数
            checkpoint = {'epoch': epoch, 'state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict(), "acc": acc, "acc2": acc2, "loss_": loss_}
            model.save_checkpoint(checkpoint, t2=t,dr2=dr)
            print(t)
        if isPlot:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(loss_)
            ax1.set_ylabel('LOSS')
            ax1.set_title("lr:" + str(t[0]) + ",weight_decay:" + str(t[1]))
            ax1.set_xlabel('epoch')

            ax2 = ax1.twinx()  # this is the important function
            ax2.plot(acc)
            ax2.set_xlim([0, epoch ])
            ax2.set_ylabel('ACC')
            plt.savefig(dr+str(t[0]) + "_" + str(t[1]) + '_new.jpg')
            plt.clf()

            l1, = plt.plot(np.squeeze(acc), "b", lw=1)
            l3, = plt.plot(np.squeeze(acc2), "r")
            plt.legend([l1,  l3], ['Train_acc', 'Test_acc'], loc='lower right')
            # plt.legend([l1, l2], ['Train_acc', 'loss'], loc='upper right')
            plt.xlabel('iterations')
            plt.title("learning_rate=" + str(t[0]) + "  weight_decay=" + str(t[1]))
            plt.savefig(dr+str(t[0]) + "_" + str(t[1]) + '_2_new.jpg')
            plt.clf()

        df.loc[str(t), '训练集正确率'] = b
        df.loc[str(t), '测试集正确率'] = a

        endtime = time.time()
        dtime = endtime - starttime
        df.loc[str(t), '耗时'] = dtime
        print(str(t)+"时，耗时：%.8s s " %dtime)
        print(dr)
        print(df)
        torch.cuda.empty_cache()
