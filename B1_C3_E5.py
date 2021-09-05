import urllib.request
import zipfile

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

# class_mode is categorical given more than 2 subdirectories
train_generator = training_datagen.flow_from_directory(
    training_dir,
    target_size=(150,150),
    class_mode='categorical'
    )

