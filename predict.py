import tensorflow as tf
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from PIL import Image 

from keras.models import load_model
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

img = cv2.imread('MNIST_IMAGE.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
ret, img = cv2.threshold(img,127,255,0)
img = cv2.resize(img, (28,28))
im2arr = img.reshape(1,28,28,1)
img = im2arr.astype('float32')
img /= 255
model = load_model("final_model.h5")
y_pred = model.predict(img)
print(y_pred.argmax())
