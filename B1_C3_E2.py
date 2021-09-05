import urllib.request
import zipfile

url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
file_name = "horse-or-human.zip"

#to use Keras.ImageDataGenerator later, ensure the local directory structure has a set of named subdirectory (used as label)
training_dir = 'horse-or-human/training/'

#download the ZIP of the training data and unzip it into named directory
urllib.request.urlretrieve(url, file_name)
zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#crate an instance of ImageDataGenerator called train_datagen & all images will be rescaled by 1./255
#with image augmentation (the elements after rescale): create additional new data by amending exisiting ones by using a number of transforms
train_datagen = ImageDataGenerator(
    #normalize the image pixel to [0,1] to ensure activation on neurons
    rescale=1/255,
    #rotating each image randomly up to 40 degree left or right
    rotation_range=40,
    #shifting the image up to 20% horizontally
    width_shift_range=0.2,
    #shifting the image by up to 20% vertically
    height_shift_range=0.2,
    #shearing the image by up to 20% (i.e. shift y (x) while x (y) stay the same)
    shear_range=0.2,
    #zooming the image by up to 20% (number=scale-up both width and height, <1 is zoom-in and >1 is zoom-out; list=[width_zoom_range, height_zoom_range])
    zoom_range=0.2,
    #randomly flipping the image horizontally (vertical=vertical_flip)
    horizontal_flip=True,
    #filling in any missing pixels after a move or shear with nearest neighbours
    fill_mode='nearest'
    )
#specifications: binary=2 types of image; categorical=if more than two
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='binary'
    )

#model architecture:
#1st conv layer: 16 filters with each of 3x3 filter size, input shape is 300x300 pixel with color (=3)
#                300x300--> (300-2)x(300-2); #para=16x((3x3)x3color+1 bias)=16x28=448
#                several smaller conv layers as image source is large
#2nd maxpooling layer: 298x298-->298/2x298/2 =149x149; # para=0
#3rd conv layer: 149x149-->(149-2)x(149-2)=147x147; #para=32 conv x previous 16 conv x (3x3 para for each filter to learn) +32x1 bias=4640
#4th maxpooling layer: 147x147-->rounddown(147/2)xroundown(147/2)=73x73; #para=0
#5th conv layer: 73x73-->71x71; #para=64 conv x previous 32 conv x (3x3 para) + 64 x 1bias=18496
#6th maxpooling layer: 71x71-->35x35
#7th conv layer: 35x35-->33x33; #para=64x64x9+64x1=36928
#8th maxpooling layer: 33x33-->16x16
#9th conv layer: 16x16-->14x14; #para=64x64x9+64x1=36928
#10th maxpooling layer: 14x14-->7x7
# after all conv and pooling layers,the image end up with simpler 7x7 feature maps, and then they will be passed to dense layer to match the appropriate labels
#11th flatten layer: after covolutions and pooling, prior to dense layers, the data will be flattened from image value to numeric value
#12th dense layer: hyperparameter tunning with 512 neurons; #para=512x64x(7x7)+512=1606144
#13th desne layer: output layer, dense(1, activation='sigmoid'): we can get binary classification with just a single neuron if we activate it with sigmoid
                   #para=1x512+1=513
# total #para=448+4640+18496+36928+36928+1606144+513=1,704,097
import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',
    input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

#loss func=binary_crossentropy for binary results
#optimizer=RMSprop:root mean square propagation with learning rate =0.001
model.compile(loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    metrics=['accuracy'])

#add validation dataset on top of training dataset
#download validation dataset and unzip it into a validation directory
validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
validation_file_name = "validation-horse-or-human.zip"
validation_dir = 'horse-or-human/validation/'
urllib.request.urlretrieve(validation_url, validation_file_name)
zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()

#create an instance of ImageDataGenerator called validation_datagen
validation_datagen = ImageDataGenerator(rescale=1/255)
#specifications of validation_datagen
validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode='binary'
    )

#use fit_generator to train model (not model.fit due to use of generator) and pass it to train_generator we firstly created
#and use the validation_data to test the model
history = model.fit_generator(
    train_generator,
    epochs=15,
    validation_data=validation_generator
    )

#pre-augmentation result: it showed ~100% accuracy on training set while ~80% on validation set, indicating overfitting
#post-augmentation result: it showed ~95% accuracy on training set and ~91% on validation set on the 14th epoch