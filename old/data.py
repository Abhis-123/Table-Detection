import tensorflow as tf

img_height, img_width = 256, 256

def normalize(input_image):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  return input_image


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



def readimage(file_path):
    file_path = tf.strings.regex_replace(file_path, '.xml', '.PNG')
    file_path = tf.strings.regex_replace(file_path, 'labels', 'images')
    img = normalize(decode_img(tf.io.read_file(file_path)))
    print(file_path)
    return img
def readmask(file_path):
    file_path = tf.strings.regex_replace(file_path, '.xml', '.jpeg')
    file_path = tf.strings.regex_replace(file_path, 'labels', 'tablemask')
    img = normalize(decode_mask_img(tf.io.read_file(file_path)))    
    print(file_path)
    return img
def readpath(file_path):
    img=p(file_path)
    tbmask=q(file_path)

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'Table Mask' 
  ]

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()




dataset=list_ds.map(r)


DATASET_SIZE = len(list(list_ds))
train_size = int(0.8 * DATASET_SIZE)
test_size = int(0.2* DATASET_SIZE)
#train_size=250
train = dataset.take(train_size)
test = dataset.skip(train_size)
TRAIN_LENGTH = len(list(train))
BUFFER_SIZE = 1000