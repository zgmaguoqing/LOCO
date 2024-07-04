from zhiyuan import OWMLayer
import torch
from hparams import HyperParams as hp
from spikingjelly.clock_driven import neuron,encoding
class clusterOWMLayer(OWMLayer):
    '''
    cluster要维护两个数据结构。一个是输入量，一个是聚类均值。
    输入量在每次输入时添加。为简化问题，可以记录输入量编号。共维护batchsize个输入量。
    聚类均值在每次刷新输入量时重新计算。
    '''
    def __init__(self,in_features: int,
                 out_features: int,
                 bias: bool = True,
                 ** kwargs
                 ):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias
                         )
        self.input_list=None #以扩展1为格式存储的输入向量
        self.dirs=None
        self.clust_method=kwargs['clust_method'](max_iter=10,verbose=False,device=hp.device)

        self.T = 500
        self.encoder = encoding.PoissonEncoder()
        self.if_node = neuron.IFNode(v_reset=None, )  # monitor_state=True

    def OWP(self,loss,x): #根据本层的投影矩阵，修饰梯度。根据loss选择要固定的方向
        '''

        :param loss: loss代表与真值的距离
        :param x:
        :return:
        '''
        with torch.no_grad():
            inputs=x
            x=x.mean(dim=0).unsqueeze(dim=0)
            deleted_dirs=self.delete_dre(self.input_list,x)

            # 根据锁定的方向计算投影矩阵
            self.owm_P = self.calcu_owm_P(deleted_dirs)
            # self.Proj(self.P)

            ##### LowDim projection
            inputs = torch.cat([inputs, torch.ones(inputs.shape[0]).unsqueeze(dim=1)], dim=1)
            sorted_maindirections = self.pca(inputs)

            owm_maindirections = self.pit_protected_dirs(self.owm_P, sorted_maindirections)
            # 根据锁定的方向计算投影矩阵
            self.LowDim_P = self.calcu_LowDim_P(owm_maindirections)





            # self.P=self.LowDim_P-self.owm_P
            self.Proj(self.LowDim_P)

    def delete_dre(self,input_list,x):
        '''
        1.K-means计算聚类均值(input_list->dires)
        2.通过x查找最近的聚类均值，将其从锁定的子空间方向中删除。
        :param x:
        :return:
        '''
        dirs=self.clust_method.fit(input_list,hp.n_kmeans)

        deleted_dirs=self.delete_nearest_dre(dirs,x)

        # index=None
        # in_dire=torch.cat([x, torch.ones(x.shape[0]).unsqueeze(dim=0)],dim=1)[0]
        # for i in range(dirs.shape[1]):
        #     if torch.equal(dirs[:,i],in_dire):
        #         index=i
        #         break
        #
        # assert index is not None
        #
        # # 删除第index个方向
        # deleted_dirs = dirs[:,torch.arange(dirs.size(1)) != index]
        return deleted_dirs

    def add_dir(self,input):
        '''
        添加输入量到输入量数据结构，刷新输入量数据结构
        输入量数据结构大小为batchsize。
        输入量数据结构模长应当归一化。因为owm需要对方向角进行聚类，应当忽略模长信息。为了简便起见，使用模长归一化的方法，后对向量进行聚类。
        这样可以直接套用kmeans根据欧氏距离聚类的方法聚类。写起来简便。更严格的方法是kmeans距离度量应该是角度度量，算出的均值应当也进行模长归一化。
        :return:
        '''
        self.count += 1
        # input shape: [batch,dim]
        dire = input.mean(dim=0).unsqueeze(dim=1).detach()
        dire = torch.cat([dire, torch.ones(dire.shape[1]).unsqueeze(dim=0)])  # 列排列
        dire=dire/torch.sqrt(torch.pow(dire,2).sum(dim=0))# 模长归一化

        if self.input_list is not None and self.input_list.shape[1] < hp.n_data:
            self.input_list = torch.cat([self.input_list, dire], dim=1)
        elif self.input_list is not None and self.input_list.shape[1] == hp.n_data:
            # self.dirs=self.replace_dire(self.dirs,dire)
            self.input_list[:, (self.count - 1) % int(hp.n_data)] = dire[:, 0]
        elif self.input_list is None:
            self.input_list = dire

    def delete_nearest_dre(self,dirs,x):
        in_dire = torch.cat([x, torch.ones(x.shape[0]).unsqueeze(dim=0)], dim=1)[0]
        if dirs is None:
            return torch.zeros((x.shape[1]+1,0))

        # find mini_theta dires
        min_theta = 90
        similar_index = None
        for i in range(dirs.shape[1]):
            theta = self.calcu_theta(dirs[:, i], in_dire)
            if min_theta > theta:
                min_theta = theta
                similar_index = i

        deleted_dirs =dirs[:, torch.arange(dirs.size(1)) != similar_index]

        return deleted_dirs

    def encode(self,x):
        for i in range(self.T):
            encoded_x=self.encoder(x).unsqueeze(dim=0)
            try:
                spike_x=torch.cat((spike_x,encoded_x),dim=0)
            except:
                spike_x = self.encoder(x).unsqueeze(dim=0)
        spike_x=spike_x.permute(1,2,0)#torch.transpose(spike_x,0,-1)
        return spike_x

    def snn_weight_propagetion(self,x):
        neural_inspikes=torch.zeros(x.shape[0],self.out_features,x.shape[2])
        for i in range(self.T):
            neural_inspikes[:,:,i]=self(x[:,:,i])

        return neural_inspikes
    def snn_neuron_propagation(self,neural_inspikes):
        neural_outspikes = torch.zeros_like(neural_inspikes)
        self.if_node.reset()
        for i in range(self.T):
            neural_outspikes[:, :, i] = self.if_node(neural_inspikes[:, :, i])
        return neural_outspikes

    def decode(self,neural_outspikes):
        y=neural_outspikes.mean(dim=-1)
        return y

    def calcu_LowDim_P(self, owm_maindirections):
        dire=owm_maindirections#sorted_maindirections[:hp.n_pcas+hp.n_kmeans].T
        dire = dire.detach()
        U = dire  # A[:, torch.arange(A.size(1)) != j]  # U是unchange space
        M = torch.mm(U.T, U)

        M = torch.inverse(M)
        # M=torch.tensor(1/np.array(M))

        P = torch.mm(U, M)
        P = torch.mm(P, U.T)
        # P = torch.eye(dire.shape[0]) - P

        return P
    def calcu_owm_P(self,dire):
        dire=dire.detach()
        U = dire#A[:, torch.arange(A.size(1)) != j]  # U是unchange space
        M = torch.mm(U.T, U)

        M = torch.inverse(M)
        #M=torch.tensor(1/np.array(M))

        P = torch.mm(U, M)
        P = torch.mm(P, U.T)
        P = torch.eye(dire.shape[0]) - P

        return P


    def pca(self,x):
        mean = torch.mean(x, dim=0)
        x_centered = x - mean
        cov_matrix = torch.matmul(x_centered.T, x_centered) / (x_centered.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
        sorted_indices = torch.argsort(eigenvalues.real, descending=True)
        sorted_eigenvectors = eigenvectors[:, sorted_indices].real.permute(1, 0)
        # 将均值和排序后的主成分拼接在一起
        sorted_maindirections = torch.cat((mean.unsqueeze(0), sorted_eigenvectors), dim=0)
        return sorted_maindirections

    def pit_protected_dirs(self,owm_P, sorted_maindirections):
        maindirections=sorted_maindirections[:hp.n_pcas+hp.n_kmeans].T
        # indexs=[]
        # for i in range(maindirections.shape[1]):
        #     origin_dire=maindirections[:,i:i+1]
        #     projected_dire=torch.mm(owm_P,origin_dire)
        #     if self.same_dire(origin_dire,projected_dire):
        #         indexs.append(i)
        # for i in range(maindirections.shape[1]):
        #     maindirections[:,i:i+1]=torch.mm(owm_P,maindirections[:,i:i+1])
        owm_maindirections=torch.mm(owm_P,maindirections)
        owm_maindirections=self.gram_schmidt(owm_maindirections)
        return owm_maindirections#[:,indexs]
    def same_dire(self,origin_dire,projected_dire):
        mode_origin=torch.mm(origin_dire.T,origin_dire)
        mode_projected = torch.mm(projected_dire.T, projected_dire)
        if mode_projected<0.9*mode_origin:
            return False
        else:
            return True

    def gram_schmidt(self,A):
        n = A.size(1)  # 获取列的数量
        Q = torch.zeros_like(A)
        indexs=[]
        for i in range(n):
            # 初始化为原向量
            v = A[:, i]

            # 减去已经正交化的向量部分
            for j in range(i):
                q = Q[:, j]
                v = v - (torch.dot(v, q) / torch.dot(q, q)) * q

            # 正交化
            Q[:, i] = v  # / torch.norm(v)
            if torch.norm(v)>0.1*torch.norm(A[:, i]):
                indexs.append(i)
            else:
                a=1
                pass

        return Q[:,indexs]