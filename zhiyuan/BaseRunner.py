import torch
import numpy as np
class BaseRunner:
    def __init__(self,data_loader,net,visualizer):
        self.data_loader=data_loader
        self.net=net
        self.visualizer=visualizer

    def run(self,epoch):
        self.train(epoch=epoch)
        self.test(rende=True)

    def train(self,epoch):
        train_data=self.data_loader.train_data #train_data可能只有输入，也可能有输入和target
        for i in range(epoch):
            loss = self.net.train(train_data)  # 验证集准确率
            if i % 100 == 0:
                error = self.test(rende=False)  # 测试集准确率
                print('error=', error)
                torch.save(self.net, 'PActornet')

    def test(self,rende=True):
        test_data = self.data_loader.test_data
        error, y = self.net.test(test_data)

        if rende:
            self.visualizer.render(test_data, y)
        return error

