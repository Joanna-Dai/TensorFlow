# Step 1: preprocessing
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
training_size=23000

training_sentences=sentences[0:training_size]
testing_sentences=sentences[training_size:]
training_labels=labels[0:training_size]
testing_labels=labels[training_size:]

# create word index based on training set and use it for testing set
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
max_length = 10
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, padding=padding_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, padding=padding_type)

# convert training and testing set from array to numpy format for tensorflow training
import numpy as np

training_padded = np.array(training_padded)
testing_padded = np.array(testing_padded)
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)


# Step 2: embeddings

import tensorflow as tf

# tf.keras.layers.Embedding(vocab_size, embedding_dim): initialize embedding layer with vocab size and embedding dimensions, an array with embedding_dim for each word
# the dimensions will be learned through backpropagation as the network learns by matching the training data to its labels
embedding_dim=16
# model architecture
model=tf.keras.Sequential([
    # initialize embedding layer: every word in 10000 vocab size will be assigned 16 dimensional vectors
    # note: vocab_size in tokenization stage and embedding stage should be aligned
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    # feed the output embedding layer into a dense layer by using pooling
    # GlobalAveragePoooling: the dimensions of the embeddings are averaged out to produce a fixed-length output vector
    # 0 trainable parameters
    tf.keras.layers.GlobalAveragePooling1D(),
    # 24-neuron dense layer: hyperparameter tunning to learn #para=24-neuron x (16-dimension +1-bias) = 24x17=408
    tf.keras.layers.Dense(24, activation='relu'),
    # single-neuron dense layer for binary classification: #para=1-neuron x 24-output of previous layer + 1-bias=25
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#check num of para
model.summary()

#result after 30 epochs: training accuracy=98%, validation accuracy=76% (likely due to validation data contains words that aren't present in training data)
#       indicators of overfitting: over the time, validation accuracy drops a little bit but validation loss increases sharply
#       (i.e. the neural network is good at matching patterns in 'noisy' data in the training set that doesn't exist anywhere else. The better it gets at matching it, the worse the loss of the validation set will be.)
num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)




