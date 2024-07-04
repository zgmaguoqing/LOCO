import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
class Visualizer:
    def __init__(self):
        self.t=0

    def render(self,x,y):
        self.t += 1
        if self.t == 1:
            plt.ion()
            plt.figure(figsize=(3, 2))
            self.ax = Axes3D(plt.gcf())
        z, t = x
        standered_data = np.hstack([t, z]).T
        self.IC_plot(y.T[0].detach(), y.T[1].detach(), z[:, 0], standered_data)

    def IC_plot(self,*in1):
        self.plot_3D(*(*in1,self.ax))

    def plot_3D(self,x, y, z, standered_data, ax):
        # 设置画布的大小

        plt.cla()
        ax.plot3D(standered_data[0], standered_data[1], standered_data[2], c='#DC143C')
        ax.plot3D(x.detach().numpy(), y.detach().numpy(), z, c='#00CED1')
        # plt.show()
        plt.pause(0.1)