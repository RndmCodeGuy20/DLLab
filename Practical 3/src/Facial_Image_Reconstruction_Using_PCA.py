import os
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


class FacialImageCompressionUsingPCA:

    def __init__(self):
        self.facesList = os.listdir("D:\DLLab\Practical 3\data\lfwcrop_grey/faces")
        self.X = []

    def CreateXMatrix(self):
        for face in self.facesList[:10]:
            image = cv.imread(f"D:/DLLab/Practical 3/data/lfwcrop_grey/faces/{face}", 0)
            print(image.shape)
            self.X.append(np.ravel(np.array(image)))

            # self.X.append(cv.imread(f"D:/DLLab/Practical 3/data/lfwcrop_grey/faces/{face}", 0))
        self.X = np.array(self.X)
        print(self.X)

    def GetMeanFace(self):
        meanFace = np.mean(self.X, axis=0)
        # print(meanFace.shape)

        plt.imshow(meanFace, cmap='gray')
        plt.show()

    def GetEigenFaces(self):
        XTX = self.X.T.dot(self.X)
        eigenvalues, eigenvectors = np.linalg.eig(XTX)

        print(f"eigenvalues = {eigenvalues}, eigenvectors = {eigenvectors.shape}")

        # image = np.array(cv.imread(f"D:/DLLab/Practical 3/data/lfwcrop_grey/faces/Aaron_Eckhart_0001.pgm", 0))
        image = np.zeros((64, 64))
        for i in eigenvectors:
            image += i.reshape(64, 64).dot(self.X[0].reshape(64, 64).T).dot(i.reshape(64, 64))

        plt.imshow(image, cmap='gray')
        plt.show()

    def PCA(self):
        print(self.X[0])
