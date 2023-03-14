import os
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


class FacialImageCompression:
    Feature_Matrix = np.array([])

    def __init__(self):
        pass

    def ShowImage(self):
        images = []
        for i in os.listdir('../data/lfwcrop_grey/faces/')[:10]:
            image = np.array(Image.open(f'../data/lfwcrop_grey/faces/{i}'))
            images.append(image)

        images = np.array(images)

        self.Feature_Matrix = images.reshape(10, 64 * 64)
        print(self.Feature_Matrix.shape, images.shape)

        plt.imshow(images[0], cmap='gray')
        plt.show()

    def GetEigenVectors(self):
        cov_matrix = np.cov(self.Feature_Matrix.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        eigen_pair = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]

        # Sort the pairs according to decreasing eigenvalues
        eigen_pair.sort(key=lambda x: x[0], reverse=True)

        # print(eigen_pair)

        keep_variance = 0.99

        required_variance = keep_variance * sum(eigenvalues)

        required_dim = 0
        variance = 0
        for i in range(len(eigen_pair)):
            variance += eigen_pair[i][0]
            if variance >= required_variance:
                required_dim = i + 1
                break

        print('Total Dimensions: {}'.format(len(eigen_pair)))
        print('Required Dimensions: {}'.format(required_dim))

        projection_matrix = np.empty(shape=(self.Feature_Matrix.shape[1], required_dim))

        for index in range(required_dim):
            eigenvector = eigen_pair[index][1]
            projection_matrix[:, index] = eigenvector

        print('Projection Matrix Shape: \n {}'.format(projection_matrix.shape))
        basis = projection_matrix.reshape(64, 64, required_dim)

        plt.figure(figsize=(20, 30))

        plt.imshow(basis[:, :, 0])
