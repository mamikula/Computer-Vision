import cv2
import matplotlib.pyplot as plt

imgpath = "img_in.png"
img = cv2.imread(imgpath)

cv2.imshow('Image', img)
cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Color Channels Histogram
red_hist = cv2.calcHist([img], [0], None, [256], [0, 255])
green_hist = cv2.calcHist([img], [1], None, [256], [0, 255])
blue_hist = cv2.calcHist([img], [2], None, [256], [0, 255])

plt.subplot(4, 1, 1)
plt.imshow(img)
plt.title('image')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 1, 2)
plt.plot(red_hist, color='r')
plt.xlim([0, 255])
plt.title('red histogram')

plt.subplot(4, 1, 3)
plt.plot(green_hist, color='g')
plt.xlim([0, 255])
plt.title('green histogram')

plt.subplot(4, 1, 4)
plt.plot(blue_hist, color='b')
plt.xlim([0, 255])
plt.title('blue histogram')

plt.tight_layout()
plt.show()