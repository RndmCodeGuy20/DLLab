import numpy as np


class GradientDescentFamily:
    w = 0.0
    b = 0.0
    velocity = 0.0

    X = np.array([])
    Y = np.array([])

    lr = 0.0
    gamma = 0.0

    def __init__(self, w, b, lr, gamma):
        self.w = w
        self.b = b

        self.lr = lr
        self.gamma = gamma

        self.X = np.array([1, 3.5, 6])
        self.Y = np.array([4, 6.5, 9])

    def Get_Gradient(self):
        dldw = 0.0
        dldb = 0.0

        N = self.X.shape[0]

        dldw += -2 * self.X * (self.Y - (self.w * self.X + self.b))
        dldb += -2 * (self.Y - (self.w * self.X + self.b))

        # for xi, yi in zip(self.X, self.Y):
        #     dldw += -2 * xi * (yi - (self.w * xi + self.b))
        #     dldb += -2 * (yi - (self.w * xi + self.b))

        # self.w -= self.lr * np.sum(dldw) / N
        # self.b -= self.lr * np.sum(dldb) / N
        #
        # return self.w, self.b

        return dldw, dldb

    def Vanilla_Gradient_Descent(self):
        for epoch in range(300):
            dldw, dldb = self.Get_Gradient()
            # print(dldw, dldb, self.w, self.b)

            self.w -= self.lr * np.sum(dldw) / self.X.shape[0]
            self.b -= self.lr * np.sum(dldb) / self.X.shape[0]

        print(f'w: {self.w}, b : {self.b}')

    def Momentum_Gradient_Descent(self):
        dldw, dldb = self.Get_Gradient()

        for epoch in range(300):
            dldw, dldb = self.Get_Gradient()
            # print(dldw, dldb, self.w, self.b)

            self.velocity = self.gamma * self.velocity + self.lr * np.sum(dldw) / self.X.shape[0]

            self.w -= self.velocity

            # self.w -= self.lr * np.sum(dldw) / self.X.shape[0]
            self.b -= self.lr * np.sum(dldb) / self.X.shape[0]

        print(f'w: {self.w}, b : {self.b}')


if __name__ == '__main__':
    gd: GradientDescentFamily = GradientDescentFamily(0.0, 0.0, 0.05, 0.001)

    gd.Vanilla_Gradient_Descent()
    gd.Momentum_Gradient_Descent()
    # for i in range(200):
    #     dldw, dldb = self.Get_Gradient()
    #     print(dldw, dldb, self.w, self.b)
    #
    #     self.w -= self.lr * np.sum(dldw) / self.X.shape[0]
    #     self.b -= self.lr * np.sum(dldb) / self.X.shape[0]
