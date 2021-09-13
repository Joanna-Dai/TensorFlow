import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

# load dataset of a specific version: data, info = tfds.load("cnn_dailymail:3.1.0", with_info=True)
data = tfds.load('horses_or_humans', split='train', as_supervised=True)

# instead of train_batches = data.shuffle(100).batch(10), map data first with mapping function
def augmentimages(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/255)
    # random flip left or right of the image. more functions are available in tf.image for augmentation
    image = tf.image.random_flip_left_right(image)
    # some func in ImageDataGenerator augmentaiton (e.g. rotate) can only be found in tfa library
    # rotating each image randomly up to 40 degree left or right and filling in missing pixel with nearest neighbours
    image=tfa.image.rotate(image,40, interpolation='NEAREST')
    return image, label

train=data.map(augmentimages)
train_batches=train.shuffle(100).batch(32)