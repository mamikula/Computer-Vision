# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from tensorflow.keras.utils import  plot_model
import matplotlib.pyplot as plt

BATCH_SIZE = 64
EPOCH_SIZE = 64

print('tf.__version__={}'.format(tf.__version__))               # 2.17
print('tf.keras.__version__={}'.format(tf.keras.__version__))         

'''
detect synthetic white shape over black background
'''

# transfer learning - load pre-trained vgg and replace its head
vgg = tf.keras.applications.VGG16(input_shape=[128, 128, 3], include_top=False, weights='imagenet')

# freeze the base model
vgg.trainable=False

for layer in vgg.layers:
    #print( layer.name, layer.trainable)   
    if layer.name in 'block5_conv3':
       print( layer.name, layer.trainable)     
       layer.trainable = True
       print( layer.name, layer.trainable)          

x = Flatten()(vgg.output)
print('vgg.output.shape={}'.format(vgg.output.shape))
x = Dense(3, activation='sigmoid')(x)
model1 = Model(vgg.input, x)
#print(model1.summary())
#print('\n')

model1.compile(loss='binary_crossentropy', optimizer=Adam(0.001))

# plot the model
plot_model(model1, "first_model.png",show_shapes=True,expand_nested=False)


"""create a circle-generator"""

from matplotlib.patches import Circle

def synthetic_gen(batch_size=64):
  # enable generating infinite amount of batches
  while True:
      # generate black images in the wanted size
      X = np.zeros((batch_size, 128, 128, 3))
      Y = np.zeros((batch_size, 3))
      # fill each image
      for i in range(batch_size):
        x = np.random.randint(8,120)
        y = np.random.randint(8,120)
        a = min(128 - max(x,y), min(x,y))
        r = np.random.randint(4,a)
        for x_i in range(128):
          for y_i in range(128):
            if ((x_i - x)**2) + ((y_i - y)**2) < r**2:
              X[i, x_i, y_i,:] = 1
        Y[i,0] = (x-r)/128.
        Y[i,1] = (y-r)/128.
        Y[i,2] = 2*r / 128.
      yield X, Y

# sanity check - plot the images
x,y = next(synthetic_gen())
plt.imshow(x[0])
plt.show()

x,y = next(synthetic_gen())
plt.imshow(x[0])
plt.show()


"""train the model"""

# needs steps per epoch since the generator is infinite
model1.fit_generator( synthetic_gen(),steps_per_epoch=EPOCH_SIZE,epochs=5)

"""predict results"""

from matplotlib.patches import Rectangle

# given image and a label, plots the image + rectangle
def plot_pred(img,p):
  fig, ax = plt.subplots(1)
  ax.imshow(img)
  rect = Rectangle(xy=(p[1]*128,p[0]*128),width=p[2]*128, height=p[2]*128, linewidth=1,edgecolor='g',facecolor='none')
  ax.add_patch(rect)
  plt.show()


# generate new image
x, _ = next(synthetic_gen())

# predict
pred = model1.predict(x)

# examine 1 image
im = x[0]
p = pred[0]
plot_pred(im,p)

