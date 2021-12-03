import os
import cv2
import tensorflow as tf
from tablenet.tablenet import TableNet
import numpy as np
print('building mode ....')
model = TableNet.build()
model.compile()
model.load_weights('models/mymodel_485.h5')
print('model built .....')

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img,channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def decode_mask_img(img):
  # convert the compressed string to a 2D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=1)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])
def create_mask(pred_mask1):
  pred_mask1 = tf.argmax(pred_mask1, axis=-1)
  pred_mask1 = pred_mask1[..., tf.newaxis]
  return pred_mask1[0]


print('reading images')
images=os.listdir('data/images/')
img_height, img_width = 256, 256
def normalize(input_image):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  return input_image

def predict(path):
  img= normalize(decode_img(tf.io.read_file(path)))
  img=np.reshape(img,(1,256,256,3))
  mask=model.predict(img)
  mask=create_mask(mask)
  return mask.numpy()


print('writing images...')
for i in range(0,340):
  path=images[i]
  path="data/images/"+path
  image=normalize(decode_img(tf.io.read_file(path)))
  mask=model.predict(np.reshape(image,(1,256,256,3)))
  m2=create_mask(mask)
  m2=np.reshape(m2,(256,256))
  cv2.imwrite('data/masks/'+images[i],m2)


