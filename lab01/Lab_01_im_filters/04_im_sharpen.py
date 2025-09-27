import cv2
import numpy as np
 
 
#image = cv2.imread('cameraman.png')
image = cv2.imread('input-image-of-wood.jpg')
 
# Print error message if image is null
if image is None:
    print('Could not read image')
 

"""
Apply sharpening using kernel
"""
kernel3 = np.array([[0, -1,  0],
                   [-1,  5, -1],
                    [0, -1,  0]])
sharp_img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel3)
 
cv2.imshow('Original', image)
cv2.imshow('Sharpened', sharp_img)
     
cv2.waitKey()
cv2.imwrite('sharp_image.png', sharp_img)
cv2.destroyAllWindows()


