import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

fig = plt.figure(figsize=(10, 7))


# ax = plt.axes(projection="3d")


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
        dldx = 2 * x
        dldy = 2 * y

        return dldx, dldy

    # def simple_gradient(self, x, y, lr):
    #     for epoch in range(300):
    #         z = x ** 2 + y ** 2
    #
    #         del_x, del_y = self.get_gradient(x, y)
    #         x -= lr * del_x
    #         y -= lr * del_y
    #
    #         plt.scatter(x, y, z, color='b')
    #     print(x, y)

    def Plot_MeshGrid(self, x, y):
        feature_x = np.linspace(-10.0, 10.0, 101)
        feature_y = np.linspace(-10.0, 10.0, 101)

        # Creating 2-D grid of features
        [X, Y] = np.meshgrid(feature_x, feature_y)

        # fig, ax = plt.subplots(1, 1)

        Z = X ** 2 + Y ** 2

        # plots filled contour plot
        plt.contourf(X, Y, Z, cmap='Spectral')
        plt.colorbar()

    def AdaGrad(self, x: float, y: float, lr: float) -> None:
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
            x -= self.lr / (self.lr * self.eps) ** 0.5 * del_x

            self.velocity_y += del_y ** 2
            y -= self.lr / (self.lr * self.eps) ** 0.5 * del_y

            plt.scatter(x, y, z, color='b')

        print(x, y)

    def RMSProp(self, x: float, y: float) -> None:
        """

        :param x:
        :param y:
        :return: None
        """
        self.velocity_x, self.velocity_y = 0.0, 0.0
        for epoch in range(30):
            z = x ** 2 + y ** 2

            del_x, del_y = self.Get_Gradient(x, y)

            self.velocity_x = (self.beta * self.velocity_x) + ((1 - self.beta) * del_x ** 2)
            x -= self.lr / (self.lr * self.eps) ** 0.5 * del_x

            self.velocity_y = (self.beta * self.velocity_y) + ((1 - self.beta) * del_y ** 2)
            y -= self.lr / (self.lr * self.eps) ** 0.5 * del_y

            plt.scatter(x, y, z, color='y')

        print(x, y)


if __name__ == '__main__':
    gd: GradientDescentFamilyUpdated = GradientDescentFamilyUpdated(0.0, 0.0, 0.01, 0.3, 0.4)

    gd.Plot_MeshGrid(0.0, 0.0)
    # gd.AdaGrad(10, 10, 0.001)
    gd.RMSProp(10, 10)
    plt.show()
