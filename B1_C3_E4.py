import urllib.request
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# download the weights and create an instance of the Inception V3 architecture
weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)

# load the downloaded weights into the architecture
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),include_top=False,weights=None)
pre_trained_model.load_weights(weights_file)

# inspect its architecture on layers and their names:
# pre_trained_model.summary()

# freeze the entire network from retraining
for layers in pre_trained_model.layers:
    layers.trainable=False

# set a variable to point at mixed7 (nice ++++-& small at 7x7 images) output as where we want to crop the network up to
last_layer=pre_trained_model.get_layer('mixed7')
print('last layer output shape:',last_layer.output_shape)
last_output=last_layer.output

# add dense layer after last layer we get from pretrained model
# flatten the output layer to 1 dimension
x = tf.keras.layers.Flatten()(last_output)
# add a fully connected layer with 1,024 hidden units and ReLU activation
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# add a final sigmoid layer for classification
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# define the model as pretrained model's input, followed by x
model = tf.keras.Model(pre_trained_model.input, x)
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['acc'])

# use this new model with transfer learning for human-and-horse dataset
# download training dataset and validation dataset
training_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
training_file_name = "horse-or-human.zip"
training_dir = 'horse-or-human/training/'
urllib.request.urlretrieve(training_url, training_file_name)
zip_ref = zipfile.ZipFile(training_file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()

validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
validation_file_name = "validation-horse-or-human.zip"
validation_dir = 'horse-or-human/validation/'
urllib.request.urlretrieve(validation_url, validation_file_name)

zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()

# add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1.0/255.)

# flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(training_dir,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

# flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                         class_mode='binary',
                                                         target_size=(150, 150))


history = model.fit_generator(
            train_generator,
            validation_data=validation_generator,
            epochs=15)

#model result: the model (with transfer learning and augmentation) shows 99% accuracy in training set and 99% accuracy in validation set

