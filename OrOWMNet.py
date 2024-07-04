from zhiyuan import OWMNet
# from clusterOWMNet import clusterOWMNet
from clusterOWMLayer import clusterOWMLayer
from zhiyuan import OrNet
import torch
import torch.optim as optim
import torch.nn.functional as F
from hparams import HyperParams as hp

class OrOWMNet(OWMNet):
    def __init__(self,
                 n_input,
                 n_hidden_layer,
                 n_hidden,
                 n_outputs=None,
                 optimizer=optim.SGD,
                 lr=hp.lr,
                 load=True,
                 ornet=None,
                 **kwargs
                 ):
        super().__init__(n_input=n_input,
                 n_hidden_layer=n_hidden_layer,
                 n_hidden=n_hidden,
                 n_outputs=n_outputs,
                 optimizer=optimizer,
                 lr=lr,

        )
        self.fc1 = clusterOWMLayer(self.n_input, n_hidden,**kwargs)
        # self.fc1.weight.requires_grad=False
        # self.fc1.bias.requires_grad = False
        self.fc2 = clusterOWMLayer(n_hidden, n_hidden,**kwargs)
        self.fc3 = clusterOWMLayer(n_hidden, self.n_outputs,**kwargs)
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

        self.ornet=ornet # ornet正交化器的输入和OrOWMNet的输入形状一样，输出和OWMNet的输入形状一样
    def forward(self,x,mode='test'):
        x=self.ornet(x)
        # 要对x进行归一化放大，否则偏置维度始终为1，方向基本一样了。
        x=self.ornet.normal_y(x)
        x = F.one_hot(x.argmax(dim=1), num_classes=50).to(torch.float64) # onthot化，两个向量角度小于1时投影还是会出问题。先留着这个问题。
        y=super().forward(x,mode)
        return y

    def train(self,x):# 每个epoch,分batch
        '''
        每个数据点训练一次，模拟连续学习
        :param x:
        :return:
        '''
        z,t=x
        for i in range(int(z.shape[0])):
            # 每个数据点都记录前向传播的方向，后面训练时锁定前面的方向
            self.train_step(z[i:i+1], t[i:i+1])



