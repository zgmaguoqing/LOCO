import torch
import numpy as np
class BaseData_loader:
    def __init__(self,num_input,n_data):
        pass
        self.train_data=self.generate_train_data(num_input,n_data)
        self.test_data=self.generate_test_data(num_input,n_data)

    def generate_train_data(self,num_input,n_data):
        train_data=torch.rand(n_data,num_input)
        return train_data
    def generate_test_data(self,num_input,n_data):
        test_data = torch.rand(n_data, num_input)
        return test_data

class Data_loader(BaseData_loader):
    def __init__(self,n_input,n_data):
        super().__init__(num_input=n_input,n_data=n_data)


    def generate_train_data(self,num_input,n_data):
        train_data=self.data_generator(n_data)
        return train_data
    def generate_test_data(self,num_input,n_data):
        test_data = self.data_generator(n_data)
        return test_data

    def data_generator(self,n_data):  # z是自变量 ，y是因变量。y: dim*samples
        z = np.arange(0, 1, 1/n_data)
        dis = np.sin(z * 2 * np.pi) * (np.sqrt(2) / 2) + np.sqrt(2) / 2

        y = np.vstack([dis / np.sqrt(2), dis / np.sqrt(2)])
        z=np.expand_dims(z,axis=1)
        y=y.T
        return (z, y)

if __name__=='__main__':
    data_loader=Data_loader(num_input=1,
                            n_data=50)
    data_loader.generate_train_data