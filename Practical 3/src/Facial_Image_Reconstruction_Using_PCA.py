import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


class FacialImageCompression:
    def __init__(self):
        pass

    def ShowImage(self):
        # plt.imshow('../data/lfwcrop_grey/faces/Aaron_Eckhart_0001')
        image = np.array(Image.open('../data/lfwcrop_grey/faces/Aaron_Eckhart_0001.pgm'))
        plt.imshow(image, cmap='gray')

        plt.show()
