import urllib.request
import zipfile
import tensorflow as tf

# download the rock/paper/scissor training set
url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip"
file_name = "rps.zip"
urllib.request.urlretrieve(url, file_name)
# then unzip it into named directory
zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall("tmp/")
zip_ref.close()
# name training_dir by following the structure
training_dir = 'tmp/rps/'

# download the rock/paper/scissor validation set
url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip"
file_name = "rps-test-set.zip"
urllib.request.urlretrieve(url, file_name)
# then unzip it into named directory
zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall("tmp/")
zip_ref.close()
# name training_dir by following the structure
validation_dir = 'tmp/rps-test-set/'

# when using the ImageDataGenerator, the classes are loaded in alphabetical order
from tensorflow.keras.preprocessing.image import ImageDataGenerator
training_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

validation_datagen=ImageDataGenerator(
    rescale=1./255
    )

# class_mode is categorical given more than 2 subdirectories
train_generator = training_datagen.flow_from_directory(
    training_dir,
    target_size=(150,150),
    class_mode='categorical'
    )

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    class_mode='categorical'
    )

# Note the input shape is the desired size of the image:
# 150x150 with 3 bytes color
model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # activation function is softmax, which will ensure all three predictions add up to 1
    # the neurons are in alphabetical order label (i.e. [paper, rock, scissor], paper=[1,0,0])
    tf.keras.layers.Dense(3, activation='softmax')
    ])

# loss function is categorical_crossengtropy as the output is categorical (more than two classes)
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

history = model.fit(train_generator, epochs=25,validation_data = validation_generator, verbose = 1)

# multiclass classification model result: 98% accuracy for training set and 99% accuracy for validation set