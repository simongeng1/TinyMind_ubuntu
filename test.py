# -*- coding: utf-8 -*-
# @Time           : 2018/4/10 17:07
# @Author         : Barry
# @Email          : wcf.barry@foxmail.com
# @Modify Time    : 2020/4/20 16:38
# @Mender         : simon
# @Email          : simongeng1@163.com
# @os             : ubuntu
# @File           : test.py
# @Software       : PyCharm

import pandas as pd
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import model2 as model
from tqdm import tqdm
import data
from train_2 import dr,T
test1path = 'test2/'
trainpath = 'train/'

filename = os.listdir(test1path)
words = os.listdir(trainpath)   # 按时间排序 从早到晚
words = np.array(words)
testnumber = len(filename)
category_number = len(words)

net = model.net()
if torch.cuda.is_available():
    net.cuda()
net.eval()


if __name__ == '__main__':
    # for t in [[0.005, 0.05], [0.01, 0.001], [0.02, 0.01]]:
    for t in T:
        checkpoint = model.load_checkpoint(t2=t,dr2=dr)
        net.load_state_dict(checkpoint['state_dict'])

        testdatas = data.loadtestdata()
        testdatas.astype(np.float)
        n = 0
        N = 16343
        batch_size = 8
        pre = np.array([])
        batch_site = []
        while n < N:
            n += batch_size
            if n < N:
                n1 = n - batch_size
                n2 = n
            else:
                n1 = n2
                n2 = N

            batch_site.append([n1, n2])

        pred_choice = []
        for site in tqdm(batch_site):
            test_batch = testdatas[site[0]:site[1]]
            test_batch = torch.from_numpy(test_batch)
            datas = Variable(test_batch).cuda().float()
            # datas = test_batch.cuda().float()
            datas = datas.view(-1, 1, 128, 128)
            outputs = net(datas)
            outputs = outputs.cpu()
            outputs = outputs.data.numpy()
            for out in outputs:
                K = 5
                index = np.argpartition(out, -K)[-K:]
                pred_choice.append(index)
        pre = np.array(pred_choice)
        predicts = []
        for k in range(testnumber):
            index = pre[k]
            predict5 = words[index]
            predict5 = "".join(predict5)
            predicts.append(predict5)

        dataframe = pd.DataFrame({'filename': filename, 'label': predicts})
        dataframe.to_csv(dr+ str(t[0])+"_"+str(t[1])+"test_new.csv", index=False, encoding='utf-8')
        read = pd.read_csv(dr+ str(t[0])+"_"+str(t[1])+"test_new.csv")
        print(read)