import cv2
import numpy as np
 
#image = cv2.imread('cameraman.png')
image = cv2.imread('input-image-of-wood.jpg')
 
# Print error message if image is null
if image is None:
    print('Could not read image')
 
"""
Apply Median blur
"""
# medianBlur() is used to apply Median blur to image
# ksize is the kernel size
median = cv2.medianBlur(src=image, ksize=5)
 
cv2.imshow('Original', image)
cv2.imshow('Median Blurred', median)
     
cv2.waitKey()
cv2.imwrite('median_blur.png', median)
cv2.destroyAllWindows()
