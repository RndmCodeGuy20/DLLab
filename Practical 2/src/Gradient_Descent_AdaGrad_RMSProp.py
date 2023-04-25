import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits import mplot3d
import numpy as np

fig = plt.figure(figsize=(10, 7))

ax = plt.axes(projection="3d")


class GradientDescentFamilyUpdated:
    """
    :arg x: x
    :arg y: y
    :arg lr: learning rate
    :arg beta: beta
    :arg eps: epsilon
    :arg velocity_x: velocity x
    :arg velocity_y: velocity y

    :de
    """
    x = 0.0
    y = 0.0

    lr, eps = 0.0, 0.0

    velocity_x, velocity_y = 0.0, 0.0

    beta = 0.0

    def __init__(self, x, y, lr, beta, eps):
        """

        :param x:
        :param y:
        :param lr:
        :param beta:
        :param eps:
        """
        self.x = x
        self.y = y
        self.lr = lr
        self.beta = beta
        self.eps = eps

    def Objective(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        return (x - y ** 2) ** 2 + (1 - y) ** 2

    def Get_Gradient(self, x, y):
        """

        Arguments:
        ----------
        :param x:
        :param y:

        Returns:
        --------
        :returns: dldx, dldy
        """
        dldx = (x - y ** 2)
        dldy = 400 * y ** 3 - 400 * x * y + 2 * y - 2

        return dldx, dldy

    def Plot_MeshGrid(self):
        X = np.arange(-2, 3.5, 0.1, dtype=np.float64)
        Y = np.arange(-1.5, 2.0, 0.1, dtype=np.float64)

        # Creating 2-D grid of features
        X, Y = np.meshgrid(X, Y)

        Z = self.Objective(X, Y)

        # plots filled contour plot
        ax.plot_surface(X, Y, Z, cmap='twilight', edgecolor='none', alpha=0.8)
        ax.view_init(30, 600)

    def AdaGrad(self, x: float, y: float) -> None:
        """

        :param x:
        :param y:
        :param lr:
        :return: None
        """
        self.velocity_x, self.velocity_y = 0.0, 0.0
        for epoch in range(30):
            z = x ** 2 + y ** 2

            del_x, del_y = self.Get_Gradient(x, y)

            self.velocity_x += del_x ** 2
            x -= (self.lr / (self.lr * self.eps) ** 0.5) * del_x

            self.velocity_y += del_y ** 2
            y -= (self.lr / (self.lr * self.eps) ** 0.5) * del_y

            ax.scatter3D(x, y, z, color='#4f5d75', marker="*")

        print("ADAGRAD: ")
        print("final x: ", x, "final y: ", y)

    def RMSProp(self, x: float, y: float) -> None:
        """

        :param x:
        :param y:
        :return: None
        """
        self.velocity_x, self.velocity_y = 0.0, 0.0
        for epoch in range(35):
            z = x ** 2 + y ** 2

            del_x, del_y = self.Get_Gradient(x, y)

            self.velocity_x = (self.beta * self.velocity_x) + ((1 - self.beta) * del_x ** 2)
            x -= (self.lr / (self.lr * self.eps) ** 0.5) * del_x

            self.velocity_y = (self.beta * self.velocity_y) + ((1 - self.beta) * del_y ** 2)
            y -= (self.lr / (self.lr * self.eps) ** 0.5) * del_y

            ax.scatter3D(x, y, z, color='#353535', marker="o")
        print("RMSPROP: ")
        print("final x: ", x, "final y: ", y)

    def Show_Plot(self):
        # ax.legend(handles=[
        #     Patch(facecolor='#4f5d75', label="AdaGrad"),
        #     Patch(facecolor='#353535', label="RMSProp")
        # ], loc="upper right")
        # plt.legend(["AdaGrad", "RMSProp"], loc="upper right")
        plt.show()
