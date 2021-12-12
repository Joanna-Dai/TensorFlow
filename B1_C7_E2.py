# using stacking RNN layer (in this case, stacking LSTM)

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
training_size = 23000

training_sentences=sentences[0:training_size]
testing_sentences=sentences[training_size:]
training_labels=labels[0:training_size]
testing_labels=labels[training_size:]

# tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# create word index based on training set and use it for testing set
vocab_size = 20000 #adjust from 10000 to 2000
max_length = 100
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

# best practice for embedding_dim is fourth root of vocab_size, also, lower embedding_dim means faster training speed
# above rule is ignored for RNN, coz it would be too small for the recurrent layer
embedding_dim=64 # adjusted up for RNN

# model architecture
model=tf.keras.Sequential([

    # initialize embedding layer (backpropagation)
    # para#=20000*64=1,280,000
    tf.keras.layers.Embedding(vocab_size, embedding_dim),

    # the first RNN layer using bidirectional LSTM
    # neuron#=64(out)+64(back)=128
    # para#=4 x [(64-unit + 64-input + 1-bias) * 64-unit) x 4 x 2-direction=66048
    # note: all the layers prior to the last LSTM layer need "return_sequences=True"
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),

    # the second/last LSTM layer
    # stack on top of last LSTM layer just like stacking dense layers (stacking LSTMs approach is used in mny state-of-the-art NLP models)
    # para#=4 x [(64-unit + 128-input + 1-bias) * 64-unit) x 4 x 2-direction=98816
    # this extra layer gives extra 98816 parameters, an increase of (98816/p_total 1349169) 7.3%, slow network down but reasonable given benefit-cost
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),

    # 24-neuron dense layer: adjusted up for RNN
    # para#=24-neuron x (128-p_dimension + 1)= 3096
    tf.keras.layers.Dense(24, activation='relu'),

    # single-neuron dense layer for binary classification: #para=1-neuron x 24-output of previous layer + 1-bias=25
    # para#=1-neuron x (24-p_neuron + 1-bias)=25
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# change the default learning rate (0.001) for adam to mitigate overfitting (likey caused by network learns too quickly)
adam = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# optimizer='adam' is using default adam
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

#check num of para
model.summary()

# 30-epoch result: train_acc=95%, val_acc=80%, val_loss=0.43
num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)


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

# 30-epoch result: train_acc=98% val_acc=57%, val_loss=0.98 --> overfitting (model is overspecialized for the training set)

