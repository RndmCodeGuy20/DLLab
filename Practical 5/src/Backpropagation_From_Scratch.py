import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, X, Y, lr):
        self.X = X
        self.Y = Y

        data = pd.read_csv('../data/iris.csv')
