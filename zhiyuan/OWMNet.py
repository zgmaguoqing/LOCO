import torch
import torch.nn as nn
import torch.optim as optim
from hparams import HyperParams as hp
from zhiyuan import BaseNet
from zhiyuan import ProbNet
import torch.nn.functional as F
import numpy as np
from zhiyuan import OWMLayer
class OWMNet(ProbNet):
    def __init__(self,n_input,
                 n_hidden_layer,
                 n_hidden,
                 n_outputs=None,
                 optimizer=optim.SGD,
                 lr=hp.lr,
                 load=True,
                      ** kwargs
                 ):
        super().__init__(n_input=n_input,
                 n_hidden_layer=n_hidden_layer,
                 n_hidden=n_hidden,
                 n_outputs=n_outputs,
                 optimizer=optimizer,
                 lr=lr
        )

        self.fc1 = OWMLayer(self.n_input , n_hidden)
        # self.fc1.weight.requires_grad=False
        # self.fc1.bias.requires_grad = False
        self.fc2 = OWMLayer(n_hidden, n_hidden)
        self.fc3 = OWMLayer(n_hidden, self.n_outputs)
        self.fc3.weight.data.mul_(0.1)
        if load:
            try:
                loaded_actor = torch.load('PActornet')
                self.fc1 = loaded_actor.fc1
                self.fc2 = loaded_actor.fc2
                self.fc3 = loaded_actor.fc3
            except:
                print('no saved net')

        self.optim = optimizer(self.parameters(), lr=lr)

    def forward(self,x,mode='test'):
        '''
        OWMNet的特征是：前向传播时要记录每一层的输入方向；训练时要除掉自己的方向，且向以前锁定的方向的正交空间中投影。
        ProbNet遇到的问题是，每次传播的采样结果不一样。所以要考虑输入方向是记录均值方向还是采样值方向。
        理论上记录均值方向更好，这个均值是全局都不采样传播的值。但是这个值不好记录。
        对于每层神经元来说，只能知道自己当前的输入，这个输入是前面的层采样后传播而来的。
        :param x:
        :param mode:
        :return:
        '''
        if mode=='test':
            fc1_mu = torch.relu(self.fc1(x))
            fc1_mu = torch.min(fc1_mu, torch.ones_like(fc1_mu))
            fc2_mu = torch.relu(self.fc2(fc1_mu))
            fc2_mu = torch.min(fc2_mu, torch.ones_like(fc2_mu))
            fc3_mu = self.fc3(fc2_mu)
            action = fc3_mu
            # with torch.no_grad():
            #
            #     # fc1_mu = torch.relu(self.fc1(x))
            #     # fc1_mu = torch.min(fc1_mu, torch.ones_like(fc1_mu))
            #     fc1_mu = self.fc1.decode(self.fc1.snn_neuron_propagation(self.fc1.snn_weight_propagetion(self.fc1.encode(x))))
            #
            #     # fc2_mu = torch.relu(self.fc2(fc1_mu))
            #     # fc2_mu = torch.min(fc2_mu, torch.ones_like(fc2_mu))
            #
            #     fc2_mu = self.fc2.decode(
            #         self.fc2.snn_neuron_propagation(self.fc2.snn_weight_propagetion(self.fc2.encode(fc1_mu))))
            #
            #
            #     # fc3_mu = self.fc3(fc2_mu)
            #
            #     fc3_mu = self.fc3.decode(
            #         self.fc3.snn_weight_propagetion(self.fc3.encode(fc2_mu)))
            #     action = fc3_mu
        elif mode=='train':
            self.fc1_input=x
            self.fc1_mu = torch.relu(self.fc1(x))
            self.fc1_mu = torch.min(self.fc1_mu, torch.ones_like(self.fc1_mu))
            self.fc1_action_old = self.fc1_action.detach()
            self.fc1_action = torch.normal(self.fc1_mu, self.fc1_std).detach()

            self.fc2_mu = torch.relu(self.fc2(self.fc1_action))
            self.fc2_mu = torch.min(self.fc2_mu, torch.ones_like(self.fc2_mu))
            self.fc2_action_old = self.fc2_action.detach()
            self.fc2_action = torch.normal(self.fc2_mu, self.fc2_std).detach()

            self.fc3_mu = self.fc3(self.fc2_action)
            self.fc3_action = torch.normal(self.fc3_mu, self.fc3_std).detach()
            action = self.fc3_action.data.cpu().numpy()

            #OWMNet中要再前向传播一下，以记录没有扰动下的各层均值方向。
            fc1_mu = torch.relu(self.fc1(x,add_dir=True))
            fc1_mu = torch.min(fc1_mu, torch.ones_like(fc1_mu))
            fc2_mu = torch.relu(self.fc2(fc1_mu,add_dir=True))
            fc2_mu = torch.min(fc2_mu, torch.ones_like(fc2_mu))
            fc3_mu = self.fc3(fc2_mu,add_dir=True)

        return action  # 优势：相对于只有最后一层有方差的网络。每层都有方差的可以将方差调到很小，充分搜索每层的最优解。而只有最后一层有方差的方差比多层方差大很多，不利于探索。方差小了可以用大学习率


    def train_step(self,z1,t1): #each batch
        y=self(z1,mode='train') #torch.Tensor(z1,device=hp.device)
        loss,TD = self.calcu_loss(z1,y,t1)
        self.optim_step(loss,TD)

        return loss

    def optim_step(self,loss,TD):
        self.optim.zero_grad()
        # loss[0].backward(retain_graph=True)
        # loss[1].backward(retain_graph=True)
        # loss[2].backward()
        ############# layer1
        grad_wx=(self.calcu_advant(self.fc1_mu, self.fc1_std, self.fc1_action, TD) * (self.fc1_mu - self.fc1_action) * 2 * (
                    self.fc1_mu < 1) * (self.fc1_mu > 0)).to(torch.float32) # batch*n_out

        self.fc1.weight.grad=torch.mm(grad_wx.T,self.fc1_input)/grad_wx.shape[0]/256#
        self.fc1.bias.grad=grad_wx.mean(dim=0).clone()/256

        ############# layer2
        grad_wx = (self.calcu_advant(self.fc2_mu, self.fc2_std, self.fc2_action, TD) * (
                    self.fc2_mu - self.fc2_action) * 2 * (
                           self.fc2_mu < 1) * (self.fc2_mu > 0)).to(torch.float32)  # batch*n_out

        self.fc2.weight.grad = torch.mm(grad_wx.T, self.fc1_action)/grad_wx.shape[0]/256#/grad_wx.shape[0]
        self.fc2.bias.grad = grad_wx.mean(dim=0).clone()/256

        ############# layer3
        grad_wx = (self.calcu_advant(self.fc3_mu, self.fc3_std, self.fc3_action, TD) * (
                self.fc3_mu - self.fc3_action) * 2 ).to(torch.float32)  # batch*n_out

        self.fc3.weight.grad = torch.mm(grad_wx.T, self.fc2_action)/grad_wx.shape[0]/10#/grad_wx.shape[0]
        self.fc3.bias.grad = grad_wx.mean(dim=0).clone()/10


        self.fc1.OWP(loss[3],self.fc1_input)
        fc2_input = torch.relu(self.fc1(self.fc1_input))
        fc2_input = torch.min(fc2_input, torch.ones_like(fc2_input))
        self.fc2.OWP(loss[3], fc2_input)
        fc3_input = torch.relu(self.fc2(torch.relu(self.fc1(self.fc1_input))))
        fc3_input = torch.min(fc3_input, torch.ones_like(fc3_input))
        self.fc3.OWP(loss[3], fc3_input)
        self.optim.step()
        sfdsf=1

    def calcu_loss(self,z1,action,t1):

        # reward
        mu_y1 = self(z1) #torch.Tensor(z1,device=hp.device)


        TD = self.critic(action, t1) - self.critic(mu_y1.data.cpu().numpy(), t1)  # critic(y1.detach(), t1)

        fc1_loss = (self.calcu_advant(self.fc1_mu, self.fc1_std, self.fc1_action, TD) * torch.pow(
            self.fc1_mu - self.fc1_action, 2)).mean()
        fc2_loss = (self.calcu_advant(self.fc2_mu, self.fc2_std, self.fc2_action, TD) * torch.pow(
            self.fc2_mu - self.fc2_action, 2)).mean()
        fc3_loss = (self.calcu_advant(self.fc3_mu, self.fc3_std, self.fc3_action, TD) * torch.pow(
            self.fc3_mu - self.fc3_action, 2)).mean()
        loss_data = self.critic(action, t1)
        loss = [fc1_loss, fc2_loss, fc3_loss,loss_data]

        (self.calcu_advant(self.fc1_mu, self.fc1_std, self.fc1_action, TD) * (self.fc1_mu - self.fc1_action) * 2 * (
                    self.fc1_mu < 1) * (self.fc1_mu > 0))
        return loss,TD