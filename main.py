'''
输入正交化器和OWMNet组成整个网络
整个网络还是基于OWMNet类，但是前向传播的第一层使用正交化器
因此将正交化器作为OWMNet的组件，搭建新的类OrOWMNet
'''
from Data_loader import Data_loader
from zhiyuan import OrNet
from Local_or_聚类OWM import LOCO
from Kmeans import Kmeans
from zhiyuan import Visualizer
from zhiyuan import Env_setting
from Runner import Runner
import torch

from hparams import HyperParams as hp
Env_setting(seed=10)

if __name__=='__main__':
    try:
        data_loader=torch.load('./data_loader')
        # asd
    except:
        data_loader = Data_loader(n_input=hp.n_input,
                                  n_data=hp.n_data)
        torch.save(data_loader,'data_loader')

    ornet=torch.load('./OrNet')
    kmeans=Kmeans
    net = LOCO(n_input=hp.n_input,
                  n_hidden_layer=hp.n_hidden_layer,
                  n_hidden=hp.n_hidden,
                  n_outputs=hp.n_outputs,
                  optimizer=hp.optimizer,
                  lr=hp.lr,
                   ornet=ornet,
                   clust_method=kmeans,
               load=True,
                  ).to(hp.device)
    # net=torch.load('OrOWMNet')
    visualizer = Visualizer()

    runner = Runner(data_loader=data_loader,
                    net=net,
                    visualizer=visualizer
                    )
    runner.run(epoch=hp.n_epoch)




