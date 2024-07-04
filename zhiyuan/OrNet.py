import torch
import torch.nn as nn
import torch.optim as optim
from hparams import HyperParams as hp
from zhiyuan import BaseNet
import torch.nn.functional as F
import numpy as np
class OrNet(BaseNet):
    def __init__(self,
                 n_input,
                 n_hidden_layer,
                 n_hidden,
                 n_outputs=None,
                 optimizer=optim.SGD,
                 lr=hp.lr
                 ):
        super().__init__(n_input=n_input,
                 n_hidden_layer=n_hidden_layer,
                 n_hidden=n_hidden,
                 n_outputs=n_outputs,
                 optimizer=optimizer,
                 lr=lr,
                )
        self.n_input = n_input
        self.n_hidden_layer = n_hidden_layer
        self.n_hidden = n_hidden
        self.n_outputs=n_outputs

        self.fc1 = nn.Linear(self.n_input , hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, n_outputs)

        self.optim = optimizer(self.parameters(), lr=lr)

    def forward(self,x,mode='test'):
        if mode=='test':
            fc1_mu = torch.tanh(self.fc1(x))
            fc2_mu = torch.tanh(self.fc2(fc1_mu))

            fc3_mu = self.activation_function(self.fc3(fc2_mu))
            action = fc3_mu
        elif mode=='train':
            self.fc1_mu = torch.tanh(self.fc1(x))
            self.fc2_mu = torch.tanh(self.fc2(self.fc1_mu))
            fc3_mu = self.activation_function(self.fc3(self.fc2_mu))
            action = fc3_mu

        return action  # 优势：相对于只有最后一层有方差的网络。每层都有方差的可以将方差调到很小，充分搜索每层的最优解。而只有最后一层有方差的方差比多层方差大很多，不利于探索。方差小了可以用大学习率


    def train(self,x):# 每个epoch,分batch
        z,t=x

        # for j in range(2, len(z)):
        #     t1 = t[j:j+1]
        #     z1 = z[j:j+1]
        #     self.train_step(z1,t1)
        self.train_step(z, t)

    def train_step(self,z1,t1): #each batch
        y=self(torch.Tensor(z1),mode='train')
        loss = self.calcu_loss(y,t1)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss

    def test(self,test_data):
        x,target=test_data
        y = self(torch.Tensor(x), mode='train')
        onehot_y=F.one_hot(y.argmax(dim=1),num_classes=self.n_outputs)
        error=onehot_y.sum(dim=0).max()
        return error,y


    def calcu_loss(self,action,t1):

        norm_y=self.normal_y(action)
        loss=self.single_entro(norm_y)-self.total_entro(norm_y)
        # print(self.single_entro(norm_y),self.total_entro(norm_y))

        return loss


    def calcu_abs_cos(self,x1,x2):
        x1=x1.unsqueeze(dim=1)
        x2 = x2.unsqueeze(dim=1)
        cos=torch.mm(x1.T,x2)/torch.sqrt(torch.pow(x1,2))/torch.sqrt(torch.pow(x2,2))
        abs_cos=torch.abs(cos)
        return abs_cos
        pass

    def activation_function(self,x):
        a=torch.exp(-torch.pow(x,2))
        return a

    def normal_y(self,y):
        sum_y=y.sum(dim=1)
        norm_y=(y.T/sum_y).T

        return norm_y

    def single_entro(self,y): # 每条数据的熵
        single_entro=(-y*torch.log(y)).sum(dim=1)
        loss=single_entro.mean() #最小化所有单熵
        return loss

    def total_entro(self,y):
        total_y=y.sum(dim=0) # 相当于mean后*n。此项系数比较大，有利于让全体趋近于平均
        total_entro= (-total_y*torch.log(total_y)).sum(dim=0)  # 最大化总熵
        return total_entro


