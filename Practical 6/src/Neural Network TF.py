from tensorflow import keras


class NeuralNetwork:
    def __init__(self, patience, dropout):
        self.patience = patience
        self.dropout = dropout

        # nn = keras.Sequential.Dense([])
