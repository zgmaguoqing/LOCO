
from OrOWMNet import OrOWMNet
import torch
import torch.optim as optim
from hparams import HyperParams as hp
import torch.nn.functional as F
class LOCO(OrOWMNet):
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
        super(LOCO, self).__init__(
            n_input,
            n_hidden_layer,
            n_hidden,
            n_outputs=n_outputs,
            optimizer=optimizer,
            lr=lr,
            load=load,
            ornet=ornet,
            **kwargs
        )
        '''
        参数工厂传来的参数。包括模型结构，优化器及参数，是否加载模型和其他参数。
        :param n_input:
        :param n_hidden_layer:
        :param n_hidden:
        :param n_outputs:
        :param optimizer:
        :param lr:
        :param load:
        :param ornet:
        :param kwargs:
        '''
        if load:
            try:
                LOCONet = torch.load('./LOCONet')
                self.fc1.weight.data = LOCONet[0].weight.data
                self.fc1.bias.data = LOCONet[0].bias.data
                self.fc2.weight.data = LOCONet[1].weight.data
                self.fc2.bias.data = LOCONet[1].bias.data
                self.fc3.weight.data = LOCONet[2].weight.data
                self.fc3.bias.data = LOCONet[2].bias.data
                print('loaded saved net')
            except:
                print('no saved net')
    def train(self,data):
        '''
        这个版本暂时以单条数据传输。
        :param data: 元组，包含输入和标签
        :return:
        '''
        z, t = data
        z=torch.Tensor(z, device=hp.device)
        batchsize=1000
        for i in range(int(z.shape[0]/batchsize)):
            # 每个数据点都记录前向传播的方向，后面训练时锁定前面的方向
            self.train_step(z[i*batchsize:(i + 1)*batchsize], t[i*batchsize:(i + 1)*batchsize])

    def forward(self,x,mode='test'):
        # x = self.ornet(x)
        # # 要对x进行归一化放大，否则偏置维度始终为1，方向基本一样了。
        # x = self.ornet.normal_y(x)
        # x = F.one_hot(x.argmax(dim=1), num_classes=50).to(torch.float64)  # onthot化，两个向量角度小于1时投影还是会出问题。先留着这个问题。
        y =super(OrOWMNet,self).forward(x, mode)
        return y
