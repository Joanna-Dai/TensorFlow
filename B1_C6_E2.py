# using petrained embeddings (swivel here) from tensorflow hub

# split train/val sentences are same with before
# download JSON (JavaScript Object Notation) dataset
# the news headlines dataset for sarcasm detection available on Kaggle has three fields
import os
import requests

url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json"
folder_dir = "C:/Users/user/tensorflow_datasets"
file_name = "sarcasm.json"
response=requests.get(url)
with open(os.path.join(folder_dir, file_name), 'wb') as f:
    f.write(response.content)

# initial pre-porcessing to define stopwords and punctuations
from bs4 import BeautifulSoup
import string

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]

table = str.maketrans('', '', string.punctuation)

# read json file line by line, create lists of the three fields (sentences, labels, URLs) and have the setnces cleaned of unwanted words and characters
import json
with open(os.path.join(folder_dir, file_name), 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []
for item in datastore:
    sentence = item['headline'].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    sentences.append(filtered_sentence)
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# the dataset contains 26709 items
print(len(sentences))


# split the dataset into training set and testing set
training_size = 24000

training_sentences=sentences[0:training_size]
testing_sentences=sentences[training_size:]
training_labels=labels[0:training_size]
testing_labels=labels[training_size:]

# use pretrained layer from swivel:
#      this will handle tokenization, vocab management, sequencing, padding
#      this will use the words from sentences with the embeddings learned as part of swivel
#      encode sentences (not words) into a single embedding
import tensorflow as tf
import tensorflow_hub as hub

# output_shape=20 --> embedding_dim=20
hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
                           output_shape=[20], input_shape=[], dtype=tf.string, trainable=False)


# model architecture w/ hub layer
model = tf.keras.Sequential([
    # pretrained layer (alternative to embedding layer)
    hub_layer,
    # given embedding_dim=20, neuron=16 (slightly lower than 20)
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy',optimizer=adam, metrics=['accuracy'])

model.summary()

# 50-epoch result: train_acc=67%, val_acc=66%
# --> faster training with less overfitting; accuracy is low as encoding word-based embeddings into a sentence-based one while sarcastic headlines is heavily impacted by individual words
num_epochs = 50
history = model.fit(training_sentences, training_labels, epochs=num_epochs, validation_data=(testing_sentences, testing_labels), verbose=2)

# plot val_accuracy over epoch
# plot val_loss over epoch
import matplotlib.pyplot as plt

# define function to call item in history (model.fit)
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# use model to predict test sentences
test_sentences = ["granny starting to fear spiders in the garden might be real",
 "game of thrones season finale showing this sunday night",
 "TensorFlow book will be a best seller"]

predictions = model.predict(test_sentences)
print(predictions)

# Either B1_C6_E1 or B1_C6_E2, they simply treated each sentence as a bunch of words (to classify sentences).
# There was no inherent sequences invovled while the order is very important in determining the real meaning of a setence.