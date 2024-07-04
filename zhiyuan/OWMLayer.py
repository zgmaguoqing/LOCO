import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from hparams import HyperParams as hp
class OWMLayer(nn.Linear):
    def __init__(self,in_features: int,
                 out_features: int,
                 bias: bool = True,

                 ):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias
                         )
        self.in_features=in_features
        self.P=None
        self.dirs=None # 锁定的方向，按照列排列
        self.lock_index_list=hp.lock_index_list
        self.dir_index=None
        self.dir_index2 = None
        self.dir_index3 = None
        self.count=0
    def forward(self, input: Tensor,add_dir=False) -> Tensor:
        output=super().forward(input)
        if add_dir:
            self.add_dir(input)
        return output
    def add_dir(self,input):
        self.count+=1
        # input shape: [batch,dim]
        dire=input.T.detach()
        dire = torch.cat([dire, torch.ones(dire.shape[1]).unsqueeze(dim=0)]) # 列排列

        if self.lock_index_list is None:
            if self.dirs is not None and self.dirs.shape[1]<hp.n_data :
                self.dirs=torch.cat([self.dirs,dire],dim=1)
            elif self.dirs is not None and self.dirs.shape[1]==hp.n_data:
                # self.dirs=self.replace_dire(self.dirs,dire)
                self.dirs[:,(self.count-1)%int(hp.n_data)]=dire[:,0]
            elif self.dirs is None:
                self.dirs=dire
        else:
            assert len(self.lock_index_list) <hp.n_data
            index=(self.count-1)%int(hp.n_data)
            if index in self.lock_index_list:
                if self.dirs is not None and self.dirs.shape[1]<len(self.lock_index_list) :
                    self.dirs=torch.cat([self.dirs,dire],dim=1)
                elif self.dirs is not None and self.dirs.shape[1]==len(self.lock_index_list) :
                    # self.dirs=self.replace_dire(self.dirs,dire)
                    posi=self.lock_index_list.index(index)
                    self.dirs[:,posi]=dire[:,0]
                elif self.dirs is None:
                    self.dirs=dire



        pass
    def OWP(self,loss,x): #根据本层的投影矩阵，修饰梯度。根据loss选择要固定的方向
        '''

        :param loss: loss代表与真值的距离
        :param x:
        :return:
        '''

        deleted_dirs=self.delete_dre(self.dirs,x)

        # 根据锁定的方向计算投影矩阵
        self.P = self.calcu_P(deleted_dirs)
        self.Proj(self.P)

        # loss=abs(loss)
        # threshold=0.00001
        # if loss.min()<threshold and self.dir_index is  None:
        #     self.dir_index=loss.argmin()
        # if self.dir_index is not None:
        #     loss[self.dir_index]=1
        #
        # if loss.min()<threshold and self.dir_index2 is  None:
        #     self.dir_index2=loss.argmin()
        # if self.dir_index2 is not None:
        #     loss[self.dir_index2]=1
        #
        # if loss.min()<threshold and self.dir_index3 is  None:
        #     self.dir_index3=loss.argmin()
        #
        # if self.dir_index is not None :#and self.dir_index2 is None and self.dir_index3 is None:
        #     dire=x[self.dir_index:self.dir_index+1]
        #     dire=torch.cat([dire.T, torch.ones(dire.T.shape[1]).unsqueeze(dim=0)])
        #
        #     self.P=self.calcu_P(dire)
        #
        #     self.Proj(self.P)
        #     asas=1

        # elif self.dir_index is not None and self.dir_index2 is not None and self.dir_index3 is None:
        #     dire1 = x[self.dir_index:self.dir_index + 1]
        #     dire2 = x[self.dir_index2:self.dir_index2 + 1]
        #     dire=torch.cat([dire1,dire2])
        #
        #     dire = torch.cat([dire.T, torch.ones(dire.T.shape[1]).unsqueeze(dim=0)])
        #
        #     self.P = self.calcu_P(dire)
        #
        #     self.Proj(self.P)
        # elif self.dir_index is not None and self.dir_index2 is not None and self.dir_index3 is not None:
        #     dire1 = x[self.dir_index:self.dir_index + 1]
        #     dire2 = x[self.dir_index2:self.dir_index2 + 1]
        #     dire3 = x[self.dir_index3:self.dir_index3 + 1]
        #     dire=torch.cat([dire1,dire2,dire3])
        #
        #     dire = torch.cat([dire.T, torch.ones(dire.T.shape[1]).unsqueeze(dim=0)])
        #
        #     self.P = self.calcu_P(dire)
        #
        #     self.Proj(self.P)



    def delete_dre(self,dirs,x):
        '''
        两种情况：x在锁定的方向里面，x不在锁定的方向里面
        x在锁定的方向里面时，则删除对应的方向
        x不在锁定的方向里面时，
        :param x:
        :return:
        '''
        index=None
        in_dire=torch.cat([x, torch.ones(x.shape[0]).unsqueeze(dim=0)],dim=1)[0]
        for i in range(dirs.shape[1]):
            if torch.equal(dirs[:,i],in_dire):
                index=i
                break

        # assert index is not None
        if index is None:
            return dirs
        # 删除第index个方向
        deleted_dirs = dirs[:,torch.arange(dirs.size(1)) != index]
        return deleted_dirs
    def calcu_P(self,dire):
        dire=dire.detach()
        U = dire#A[:, torch.arange(A.size(1)) != j]  # U是unchange space
        M = torch.mm(U.T, U)

        M = torch.inverse(M)
        #M=torch.tensor(1/np.array(M))

        P = torch.mm(U, M)
        P = torch.mm(P, U.T)
        P = torch.eye(dire.shape[0]) - P

        return P

    def Proj(self,P):

        W = torch.cat([self.weight.grad.T, torch.unsqueeze(self.bias.grad, dim=0)])
        W = torch.mm(P, W)

        self.weight.grad = W[:self.in_features].T.clone()
        self.bias.grad = W[-1].clone()

        assert not torch.isnan(self.weight[0,0])

    def normal_dire(self,dire):
        sum_dire=dire.sum(dim=0)
        normal_dire=(dire/sum_dire)

        return normal_dire

    def replace_dire(self,dires,dire):

        # find mini_theta dires
        min_theta=90
        similar_index=None
        for i in range(dires.shape[1]):
            theta=self.calcu_theta(dires[:,i],dire[:,0])
            if min_theta>theta:
                min_theta=theta
                similar_index=i

        dires[:,similar_index]=dire[:,0]

        return dires



    def calcu_theta(self,a,b):
        theta=torch.abs((a*b).sum(dim=0))/torch.sqrt(torch.pow(a,2).sum(dim=0)*torch.pow(b,2).sum(dim=0))
        theta=torch.arccos(theta)
        return theta

