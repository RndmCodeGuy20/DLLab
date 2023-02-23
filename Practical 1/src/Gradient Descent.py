import numpy as np


class GradientDescentFamily:
    w = 0.0
    b = 0.0

    X = np.array([])
    Y = np.array([])

    lr = 0.0

    def __init__(self, w, b, lr):
        self.w = w
        self.b = b

        self.lr = lr

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

        self.w -= self.lr * np.sum(dldw) / N
        self.b -= self.lr * np.sum(dldb) / N

        return self.w, self.b

    def Vanilla_Gradient_Descent(self):
        w, b, loss = 0.0, 0.0, 0.0
        for epoch in range(600):
            w, b = self.Get_Gradient()

            y_pred = w * self.X + b
            loss = np.sum((self.Y - y_pred) ** 2) / self.X.shape[0]

        print(f'w: {w}, b : {b}, final loss: {loss}')

    def Momentum_Gradient_Descent(self):
        w, b = 0.0, 0.0
        velocity = 0.0
        for epoch in range(600):
            w, b = self.Get_Gradient()

            y_pred = w * self.X + b
            loss = np.sum((self.Y - y_pred) ** 2) / self.X.shape[0]


if __name__ == '__main__':
    gd: GradientDescentFamily = GradientDescentFamily(0.0, 0.0, 0.05)

    gd.Vanilla_Gradient_Descent()
