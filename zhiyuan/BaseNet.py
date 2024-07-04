import torch
import torch.nn as nn
import torch.optim as optim
from hparams import HyperParams as hp
class BaseNet(nn.Module):
    def __init__(self,
                 n_input,
                 n_hidden_layer,
                 n_hidden,
                 n_outputs=None,
                 optimizer=optim.SGD,
                 lr=hp.lr
                 ):
        super(BaseNet,self).__init__()
        self.n_input=n_input
        self.n_hidden_layer=n_hidden_layer
        self.n_hidden=n_hidden
        self.n_outputs=n_outputs
        self.layers=nn.ModuleList()
        if self.n_hidden_layer==0:
            self.layers.append(nn.Linear(self.n_input,self.n_outputs))
        else:

            self.layers.append(nn.Linear(self.n_input,self.n_hidden))
            for i in range(self.n_hidden_layer-1):
                self.layers.append(nn.Linear(self.n_hidden,self.n_hidden))
            if n_outputs:
                self.layers.append(nn.Linear(self.n_hidden, self.n_outputs))



        self.optim=optimizer(self.parameters(), lr=lr)
    def forward(self,x:torch.Tensor):

        if type(x)  is not torch.Tensor:
            x = torch.tensor(x)

        for i in range(self.n_hidden_layer+1):
            x=torch.tanh(self.layers[i](x))

        return x


    def train(self,train_data,batch_size=50): # 每个epoch,分batch
        input, target = train_data
        for i in range(1, int(len(input) / batch_size) + 1):
            start = (i - 1) * batch_size
            end = (i) * batch_size
            batch_input = input[start:end]
            batch_target = target[start:end]
            self.train_step(batch_input, batch_target)




    def train_step(self,input,target):
        y = self(input)
        loss = self.calcu_loss(y,target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss

    def test(self,test_data):
        input, target = test_data

        y = self(torch.Tensor(input))
        error = self.calcu_loss(y,target)
        return error,y

    def calcu_loss(self,y,target):
        loss = torch.pow(y - torch.tensor(target), 2).mean()
        return loss