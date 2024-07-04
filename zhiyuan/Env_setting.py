import numpy as np
import torch
import os
class Env_set:
    def __init__(self,seed=1):
        self.seed=seed

    def run(self):
        self.seed_torch(self.seed)
        # torch.set_default_dtype(torch.float64)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.set_device(0)
    def seed_torch(self,seed):
        # random.seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
def Env_setting(*args, **kwargs):
    Env_set(*args, **kwargs).run()