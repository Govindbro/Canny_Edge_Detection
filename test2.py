import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("source.jpg", -1)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
edges = cv2.Canny(img,100,200)

titles = ['image', 'Canny']
images = [img,edges]
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]);plt.yticks([])
plt.show()

# ABOVE IS NOW MODIFIED BY ADDING MORE FEATURES ON IT
##########################################################################################################

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# img = cv2.imread("source1.jpg", -1)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
# lap = np.uint8(np.absolute(lap))
# sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
# sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
# edges = cv2.Canny(img,100,200)

# sobelX = np.uint8(np.absolute(sobelX))
# sobelY = np.uint8(np.absolute(sobelY))

# sobelCombined = cv2.bitwise_or(sobelX, sobelY)

# titles = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined', 'Canny']

# images = [img, lap, sobelX, sobelY, sobelCombined, edges]

# for i in range(6):
#     plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]);plt.yticks([])

# plt.show()