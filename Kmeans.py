from BaseKmeans import BaseKmeans
import torch

class Kmeans(BaseKmeans):
    def __init__(self, n_clusters=None, max_iter=None, verbose=True,device = torch.device("cpu")):
        super().__init__(n_clusters=n_clusters,
                                    max_iter=max_iter,
                                    verbose=verbose,
                                    device=device,
                                    )

        self.last_means=None

    def fit(self,input_list,n):
        '''
        input_list的结构是n_input*batch
        x的结构是batch*n_input

        centers的结构是batch*n_input
        dirs的结构是n_input*batch
        :param input_list:
        :param n:
        :return:
        '''
        if input_list.shape[1]<n:
            return None
        self.n_clusters=n
        x=input_list.T

        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        # init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_row = [int(i * x.shape[0] / self.n_clusters) for i in range(self.n_clusters)]
        init_points = x[init_row]
        self.centers = init_points # centers should be some points,and shouldn't be the same.

        while self.centers.unique(dim=0).shape[0] != self.centers.shape[0]:
            self.centers =x[torch.randperm(x.shape[0])][:self.n_clusters]

        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        self.representative_sample()

        self.last_means=self.centers

        dirs=self.centers.T
        return dirs