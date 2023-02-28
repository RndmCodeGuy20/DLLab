import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('darkgrid')
fig, ax = plt.subplot(2, 3)


class GradientDescentFamily:
    w = 0.0
    b = 0.0
    velocity = 0.0
    velocity_b = 0.0

    X = np.array([])
    Y = np.array([])

    lr = 0.0
    gamma = 0.0

    N = 0

    colors = ["#9b5de5", "#f15bb5", "#fee440", "#00bbf9", "#00f5d4"]

    def __init__(self, w, b, lr, gamma):
        self.w = w
        self.b = b

        self.lr = lr
        self.gamma = gamma

        self.X = np.array([1, 3.5, 6])
        self.Y = np.array([4, 6.5, 9])

        self.N = self.X.shape[0]

    def Get_Gradient(self):
        dldw = 0.0
        dldb = 0.0

        # N = self.X.shape[0]

        dldw += -2 * self.X * (self.Y - (self.w * self.X + self.b))
        dldb += -2 * (self.Y - (self.w * self.X + self.b))

        return dldw, dldb

    def Vanilla_Gradient_Descent(self):
        loss = 0.0
        self.w = 0.0
        self.b = 0.0
        for epoch in range(300):
            dldw, dldb = self.Get_Gradient()
            # print(dldw, dldb, self.w, self.b)

            self.w -= self.lr * np.sum(dldw) / self.N
            self.b -= self.lr * np.sum(dldb) / self.N

            y_pred = self.w * self.X + self.b
            loss = np.sum((self.Y - y_pred) ** 2) / self.N

            plt.scatter(epoch, self.w, color='b')
            plt.scatter(epoch, self.b, color='r')
            # plt.plot(epoch, loss, color='g')
        plt.show()

        print(f'w: {self.w}, b : {self.b}, final loss : {loss}')

    def Momentum_Gradient_Descent(self):
        self.w = 0.0
        self.b = 0.0
        loss = 0.0

        for epoch in range(300):
            dldw, dldb = self.Get_Gradient()
            print(self.w, self.b)

            self.velocity = self.gamma * self.velocity + self.lr * np.sum(dldw) / self.N
            self.w -= self.velocity

            # self.velocity_b = self.gamma * self.velocity_b + self.lr * np.sum(dldb) / self.N
            # self.b -= self.velocity_b

            # self.w -= self.lr * np.sum(dldw) / self.X.shape[0]
            self.b -= self.lr * np.sum(dldb) / self.N

            y_pred = self.w * self.X + self.b
            loss = np.sum((self.Y - y_pred) ** 2) / self.N

        print(f'w: {self.w}, b : {self.b}, final loss : {loss}')


if __name__ == '__main__':
    gd: GradientDescentFamily = GradientDescentFamily(0.0, 0.0, 0.05, 1.009)

    gd.Vanilla_Gradient_Descent()
    gd.Momentum_Gradient_Descent()
