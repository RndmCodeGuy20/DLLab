```python
# plt.imshow('../data/lfwcrop_grey/faces/Aaron_Eckhart_0001')

        image = np.array(Image.open('../data/lfwcrop_grey/faces/Aaron_Eckhart_0001.pgm'))
        # plt.imshow(image, cmap='gray')

        print(image.shape)

        images = np.array(os.listdir('../data/lfwcrop_grey/faces'))
        # print(len(x))

        # X = images.reshape(13233, 64 * 64)
        print(images.shape)

        # plt.show()

        for i in os.listdir('../data/lfwcrop_grey/faces'):
            print(i)```