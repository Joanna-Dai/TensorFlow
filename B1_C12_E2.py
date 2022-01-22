import numpy as np
import matplotlib.pylab as plt

# Step 1: build and save the model

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


# reformat the images to be the right size for both training and inference
def format_image(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0
    return  image, label

# split dataset into train, val and test datasets; metadata is narratives info
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# num of lines
num_examples = metadata.splits['train'].num_examples
# num of unique values
num_classes = metadata.features['label'].num_classes
print(num_examples)
print(num_classes)

# shuffle the dataset with buffer_size=num_eg/4, split each window into feature image and label, batch the dataset with each batch as batch_size
BATCH_SIZE = 32
train_batches = raw_train.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = raw_validation.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = raw_test.map(format_image).batch(1)

# create feature_extractor layer, which would be the first layer of neural network
module_selection = ("mobilenet_v2", 224, 1280)
handle_base, pixels, FV_SIZE=module_selection
MODULE_HANDLE ="https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {} and output dimension {}".format(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))
feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                   input_shape=IMAGE_SIZE + (3,),
                                   output_shape=[FV_SIZE],
                                   trainable=False)

print("Building model with", MODULE_HANDLE)

# model architecture
model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

EPOCHS = 5

# train_acc=0.994, val_acc=0.988
hist = model.fit(train_batches,
                 epochs=EPOCHS,
                 validation_data=validation_batches)

# save the model out
CATS_VS_DOGS_SAVED_MODEL = "exp_saved_model"
tf.saved_model.save(model, CATS_VS_DOGS_SAVED_MODEL)


# Step 2: covert the model to TensorFlow lite

# take the saved model and convert it into .tflite model (hereby converted_model.tflite)
converter = tf.lite.TFLiteConverter.from_saved_model(CATS_VS_DOGS_SAVED_MODEL)
tflite_model = converter.convert()
tflite_model_file = 'converted_model.tflite'

with open(tflite_model_file,'wb') as f:
    f.write(tflite_model)

# after having the file, instantiate an interpretr with it
interpreter = tf.lite.Interpreter(model_path = tflite_model_file)
interpreter.allocate_tensors()

# load them into variables
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

predictions = []

# take 100 images from test_batches and test them
test_labels, test_imgs = [], []
for img, label in test_batches.take(100):
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_index))
    test_labels.append(label.numpy()[0])
    test_imgs.append(img)

# see how the predictions did against the labels
score = 0
for item in range(0,99):
    prediction = np.argmax(predictions[item])
    label = test_labels[item]
    if prediction == label:
        score = score+1
print("out of 100 predictions I got" + str(score) + " correct")

# visualize the model output
class_names = ['cat', 'dog']

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    img = np.squeeze(img)
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),

                                         class_names[true_label]), color=color)

for index in range(0,99):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(index, prediction, test_labels, test_imgs)
    plt.show()


# Step 3: Optimize/Quantize the Model

converter = tf.lite.TFLiteConverter.from_saved_model(CATS_VS_DOGS_SAVED_MODEL)
# dynamic range quantization: set the optimizations property prior to performing the conversion
# other options:
#   OPTIMIZE_FOR_SIZE --> make the model as small as possible
#   OPTIMIZE_FOR_LATENCY --> reduce inference time as much as possible
#   DEFAULT --> the best balance between size and latency
# model becomes 4x smaller with 2-3x speedup, but the accuracy dropped from 99% to 94%
converter.optimizations= [tf.lite.Optimize.DEFAULT]



tflite_model = converter.convert()
tflite_model_file = 'converted_model.tflite'

with open(tflite_model_file, "wb") as f:
    f.write(tflite_model)

