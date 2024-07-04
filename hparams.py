import torch.optim as optim
import torch
class HyperParams:
    n_data=1000

    n_input=784
    n_hidden_layer=2
    n_hidden=500
    n_outputs=10
    lock_index_list=[0]

    optimizer=optim.Adam
    lr = 0.0003/2*100#0#*0.1#*20#*1000000*0.8*2*0.01*10

    n_epoch=200000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gamma = 0.99
    lamda = 0.98
    sigma=0.005
    hidden = 64

    n_kmeans=10
    n_pcas=100

    critic_lr = 0.0003
    actor_lr = 0.0003#0.0003
    batch_size = 64
    l2_rate = 0.001
    max_kl = 0.01
    clip_param = 0.2



