import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
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
        '''

        :param num_input:
        :param n_data:
        :return: (input,target)
        input:(batchsize,n_input)
        target:(batchsize,10)
        '''
        transform_train = transforms.Compose(
            [
             # transforms.Pad(4),
             transforms.ToTensor(),
             # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
             transforms.Normalize(0,1),#(0.45,0.22),
             # transforms.RandomHorizontalFlip(),
             # transforms.RandomGrayscale(),
             # transforms.RandomCrop(32, padding=4),
             ])

        trainset = torchvision.datasets.MNIST(root='dataset_method_1', train=True, download=True,
                                                transform=transform_train)
        train_data=self.my_segmentation_transforms(trainset)
        # trainLoader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

        return train_data
    def generate_test_data(self,num_input,n_data):
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0, 1),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )

        testset = torchvision.datasets.MNIST(root='dataset_method_1', train=False, download=True,
                                               transform=transform_test)
        # testLoader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
        test_data=self.my_segmentation_transforms(testset)
        return test_data

    def data_generator(self,n_data):  # z是自变量 ，y是因变量。y: dim*samples
        z = np.arange(0, 1, 1/n_data)
        dis = np.sin(z * 2 * np.pi) * (np.sqrt(2) / 2) + np.sqrt(2) / 2

        y = np.vstack([dis / np.sqrt(2), dis / np.sqrt(2)])
        z=np.expand_dims(z,axis=1)
        y=y.T
        return (z, y)

    def my_segmentation_transforms(self,dataset):
        length=int(len(dataset))
        input_data=np.zeros((length, 28 * 28))
        target_data=np.zeros((length,10))
        indexs=dataset.targets.argsort()
        dataset.targets=dataset.targets[indexs]
        dataset.data = dataset.data[indexs]
        for index,item in enumerate(dataset):
            if index>=length:
                break
            input_data[index]=item[0].flatten()
            target_data[index]=np.eye(len(dataset.classes))[item[1]]

        data_set=(input_data,target_data)
        return data_set

class my_segmentation_transforms:
    def __init__(self,x):
        self.x=x
    def __call__(self, img, **kwargs):
        return img
if __name__=='__main__':
    data_loader=Data_loader(num_input=1,
                            n_data=50)
    data_loader.generate_train_data