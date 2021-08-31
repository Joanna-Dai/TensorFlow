#have to run on google colab or linux environment
import numpy as np
from google.colab import files
from tensorflow.keras.preprocessing import image


uploaded = files.upload()

for fn in uploaded.keys():

    # predicting images
    path = '/content/' + fn
    #load the image from Colab per path and resizes it to 300x300 to feed into model
    img = image.load_img(path, target_size=(300, 300))
    #covert the image into 2D array while the image is 3D given input_shap=(300, 300, 3)
    x = image.img_to_array(img)
    #add a new dimension to x(2D) given input_shape=(300,300,3)
    x = np.expand_dims(x, axis=0)

    #stack it vertically such that it's in the same shape with traning data
    image_tensor = np.vstack([x])

    classes = model.predict(image_tensor)

    print(classes)
    print(classes[0])
    if classes[0] > 0.5:
        print(fn + " is a human")
    else:
        print(fn + " is a horse")


# the result shows: training set cannot represent every possible scenario the model will face
#                   the model will always have some level of overspecialization toward the training set