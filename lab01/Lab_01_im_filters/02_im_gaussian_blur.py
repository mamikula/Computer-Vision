import cv2
import numpy as np
 
#image = cv2.imread('cameraman.png')
image = cv2.imread('input-image-of-wood.jpg')
 
# Print error message if image is null
if image is None:
    print('Could not read image')
 
 
"""
Apply Gaussian blur
"""
# sigmaX is Gaussian Kernel standard deviation 
# ksize is kernel size
gaussian_blur = cv2.GaussianBlur(src=image, ksize=(5,5), sigmaX=0, sigmaY=0)
 
cv2.imshow('Original', image)
cv2.imshow('Gaussian Blurred', gaussian_blur)
     
cv2.waitKey()
cv2.imwrite('gaussian_blur.png', gaussian_blur)
cv2.destroyAllWindows()

