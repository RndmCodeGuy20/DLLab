from Facial_Image_Reconstruction_Using_PCA import FacialImageCompressionUsingPCA

if __name__ == '__main__':
    fr = FacialImageCompressionUsingPCA()

    fr.CreateXMatrix()
    # fr.GetMeanFace()
    # fr.PCA()
    fr.GetEigenFaces()
