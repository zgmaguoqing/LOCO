import torch
import torch.nn as nn
import torch.optim as optim
from hparams import HyperParams as hp
from zhiyuan import BaseNet
import torch.nn.functional as F
import numpy as np
class ProbNet(BaseNet):
    def __init__(self,n_input,
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
                 lr=lr
        )
        self.n_input = n_input
        self.n_hidden_layer = n_hidden_layer
        self.n_hidden = n_hidden
        self.n_outputs=n_outputs

        self.fc1 = nn.Linear(self.n_input , n_hidden)
        logstd = torch.zeros(n_hidden)
        self.fc1_action = torch.zeros_like(logstd)
        self.fc1_std = torch.exp(logstd) * hp.sigma
        self.fc1_std_exp_avg = torch.zeros_like(self.fc1_std)
        self.fc1_std_exp_avg_sq = torch.zeros_like(self.fc1_std)
        self.fc1_std_beta1 = 0.9
        self.fc1_std_beta2 = 0.999
        self.fc1_std_lr = 1

        self.fc2 = nn.Linear(n_hidden, n_hidden)
        logstd = torch.zeros(n_hidden)
        self.fc2_action = torch.zeros_like(logstd)
        self.fc2_std = torch.exp(logstd) * hp.sigma
        self.fc2_std_exp_avg = torch.zeros_like(self.fc2_std)
        self.fc2_std_exp_avg_sq = torch.zeros_like(self.fc2_std)
        self.fc2_std_beta1 = 0.9
        self.fc2_std_beta2 = 0.999
        self.fc2_std_lr = self.fc1_std_lr


        self.fc3 = nn.Linear(n_hidden, self.n_outputs)
        logstd = torch.zeros(self.n_outputs)
        self.fc3_std = torch.exp(logstd) * hp.sigma
        self.fc3.weight.data.mul_(0.1)
        # self.fc3.bias.data.mul_(0.0)
        self.step = 0

        self.optim = optimizer(self.parameters(), lr=lr)

    def forward(self,x,mode='test'):

        if mode=='test':
            fc1_mu = torch.tanh(self.fc1(x))
            fc2_mu = torch.tanh(self.fc2(fc1_mu))
            fc3_mu = self.fc3(fc2_mu)
            action = fc3_mu
        elif mode=='train':
            self.fc1_mu = torch.tanh(self.fc1(x))
            self.fc1_action_old = self.fc1_action.detach()
            self.fc1_action = torch.normal(self.fc1_mu, self.fc1_std)

            self.fc2_mu = torch.tanh(self.fc2(self.fc1_action))
            self.fc2_action_old = self.fc2_action.detach()
            self.fc2_action = torch.normal(self.fc2_mu, self.fc2_std)

            self.fc3_mu = self.fc3(self.fc2_action)
            self.fc3_action = torch.normal(self.fc3_mu, self.fc3_std)
            action = self.fc3_action.data.numpy()

        return action  # 优势：相对于只有最后一层有方差的网络。每层都有方差的可以将方差调到很小，充分搜索每层的最优解。而只有最后一层有方差的方差比多层方差大很多，不利于探索。方差小了可以用大学习率




    def train(self,x):# 每个epoch,分batch
        z,t=x
        self.train_step(z, t)

        # for j in range(2, len(z)):
        #     t1 = t[ j,:]
        #     z1 = z[j,:]
        #

    def train_step(self,z1,t1): #each batch
        y=self(torch.Tensor(z1),mode='train')
        loss = self.calcu_loss(z1,y,t1)
        self.optim.zero_grad()
        loss[0].backward(retain_graph=True)
        loss[1].backward(retain_graph=True)
        loss[2].backward()
        self.optim.step()

        return loss

    def calcu_loss(self,z1,action,t1):

        # reward
        mu_y1 = self(torch.Tensor(z1))


        TD = self.critic(action, t1) - self.critic(mu_y1.data.numpy(), t1)  # critic(y1.detach(), t1)

        fc1_loss = (self.calcu_advant(self.fc1_mu, self.fc1_std, self.fc1_action, TD) * torch.pow(
            self.fc1_mu - self.fc1_action, 2)).mean()
        fc2_loss = (self.calcu_advant(self.fc2_mu, self.fc2_std, self.fc2_action, TD) * torch.pow(
            self.fc2_mu - self.fc2_action, 2)).mean()
        fc3_loss = (self.calcu_advant(self.fc3_mu, self.fc3_std, self.fc3_action, TD) * torch.pow(
            self.fc3_mu - self.fc3_action, 2)).mean()

        loss = [fc1_loss, fc2_loss, fc3_loss]
        return loss

    def critic(self,action, target):
        r = pow(action - target, 2).mean(axis=1)
        reward = -r
        return reward

    def calcu_advant(self,y1, std,action,TD):
        #print(std)
        TD=torch.tensor(TD)
        # P_action = torch.exp(torch.distributions.normal.Normal(y1, std).log_prob(
        #     action))  # torch.exp(torch.sum(torch.distributions.normal.Normal(y1,std).log_prob(torch.tensor(action))))
        # P_action = P_action.detach()

        advant=0.5 * TD #/torch.sqrt(torch.sqrt(P_action))
        # if abs(advant)>1:
        #     print (advant)
        return advant.unsqueeze(dim=1)