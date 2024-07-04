import torch
import numpy as np
from zhiyuan import BaseRunner
class Runner(BaseRunner):
    def __init__(self,data_loader,net,visualizer):
        super().__init__(data_loader,net,visualizer)
        # z,t = self.data_loader.train_data
        # self.net.mean_reward_of_z = {}
        # for i in z:
        #     self.net.mean_reward_of_z[str(i)] = None
    def train(self,epoch):
        x=self.data_loader.train_data #x可能只有输入，也可能有输入和target

        for i in range(epoch):
            loss=self.net.train(x)
            error,accuracy = self.test(rende=False)
            # print (error)
            if error < 0.001:
                print(i)
                while True:
                    pass
            if i % 1 == 0:

                print('error=', error,'accuracy=',accuracy)

                # torch.save([self.net.fc1,self.net.fc2,self.net.fc3,] ,'LOCONet')

    def test(self,rende=True):
        x =self.data_loader.test_data# self.data_loader.test_data
        z, t = x
        z=z[:]
        t=t[:]

        y = self.net(torch.Tensor(z))
        accuracy=(y.argmax(dim=1)==torch.tensor(t).argmax(dim=1)).sum()/y.shape[0]
        error = torch.pow(y - torch.tensor(t), 2).mean()

        if error < 0.001:
            rende=True

        if rende:
            self.visualizer.render(x,y)
        return error,accuracy