import cv2
import scipy.misc
import matplotlib.pyplot as plt

image = cv2.imread("C:/ASU/Sem 1/MWDB/Project?Hands/Hands/Hand_0000002.jpg")

image = scipy.misc.imresize(image, 0.1)

fig = plt.figure(figsize=(2, 1))
columns = 1
rows = 1

for i in range(1, columns * rows + 1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(image)

plt.show()