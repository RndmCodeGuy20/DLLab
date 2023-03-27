import numpy as np


class AutoEncoders:
    def __init__(self):
        self.xi = [1, 1, 0, 0, 1]
        self.h = [0, 0, 0, 0]
        self.xi_hat = [0, 0, 0, 0, 0]

    def InputLayer(self):
        pass

    def Encoder(self):
        for i in range(len(self.h)):
            temp = 0
            for j in range(len(self.xi)):
                temp += np.random.random() * self.xi[j]
            self.h[i] = self.EncodingFunction(temp)

        print(self.h)

    def Decoder(self):
        for i in range(len(self.xi_hat)):
            temp = 0
            for j in range(len(self.h)):
                temp += np.random.random() * self.h[j]
            self.xi_hat[i] = self.DecodingFunction(temp)

        print(self.xi_hat)

    def DecodingFunction(self, h):
        return 1 / (1 + np.exp(h))

    def EncodingFunction(self, x):
        return 1 / (1 + np.exp(x))

    def CalculateLoss(self):
        error = 0.0
        for index, val in enumerate(self.xi):
            error += (val * np.log(self.xi_hat[index]) + (1 - val) * np.log(1 - self.xi_hat[index]))

        print(error)
