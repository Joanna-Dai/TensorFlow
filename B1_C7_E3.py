# using pretrained embeddings with RNNs

# download GloVe (Global Vectors for Word Representation) model (variety of datasets)
# hereby use the dataset with 27-billion-token, 1.2-million-word vocab from a Twitter crawal of 2 billion tweets
# url = "https://github.com/stanfordnlp/GloVe" download "glove.twitter.27B.zip"

import zipfile

# unzip it into named directory
# local_zip = "C:/Users/user/tensorflow_datasets/glove.zip"
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall("C:/Users/user/tensorflow_datasets/glove")
# zip_ref.close()


import numpy as np

# create a dictionary where the key is the word and the values are the embeddings
glove_embeddings = dict()
# open and use the one of 25 dimensions
f = open("C:/Users/user/tensorflow_datasets/glove/glove.twitter.27B.25d.txt", encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embeddings[word] = coefs
f.close()

print(glove_embeddings['frog'])
print(len(glove_embeddings))

# Step 1: preprocessing sarcasm dataset as before
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
# embeddings have already been learned, vocab can be expanded
# based on the result of explore vocab_size (mostly overlapped between word_index and glove embeddings)
vocab_size = 13200
max_length = 50
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

# Step 2: create embedding matrix, for each word in tokenizer, it will use the coefficients from glove
embedding_dim = 25
# start from an i (row #) x j (col #) zero matrix
# i.e. create a matrix with the dimension of desired vocab_size and the embedding dimension
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in tokenizer.word_index.items():
    if index > vocab_size-1:
        break
    else:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


# explore vocab_size
# how many words in corpus are actually in GloVe set
xs=[] # for word_index in sarcasm dataset
ys=[] # 1 if the word in x is in glove embeddings and 0 if the word in x is not in glove embeddings
cumulative_x=[]
cumulative_y=[]
total_y=0
for word, index in tokenizer.word_index.items():
    xs.append(index) # x store the index of the word
    cumulative_x.append(index)
    if glove_embeddings.get(word) is not None: # if the word is in glove embeddings
        total_y = total_y + 1 # count how many words can be found in total
        ys.append(1) # y store 1 or 0 for the word x
    else:
        ys.append(0) # if th word is not in glove embeddings
    cumulative_y.append(total_y/index) #density about how many words can be found in glove embeddings

# plot
import matplotlib.pyplot as plt
# word frequency chart
# result: density changes somewhere between 10000 and 15000
fig, ax = plt.subplots(figsize=(12, 2))
ax.spines['top'].set_visible(False)
plt.margins(x=0, y=None, tight=True)
plt.fill(ys)
# cumulative x against cumulative y
# result: similar with above, and thereforechoose 13200 as voab_size
plt.plot(cumulative_x, cumulative_y)
plt.axis([0, 25000, .915, .985])



# Step 3: create model architecture that amend the embeddin layer to use the pretrained embeddings
import tensorflow as tf

model = tf.keras.Sequential([
    # amend embedding layer by 1) setting weights para as embedding_matrix 2) don't want the layer to be trained by trainable=False
    tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False),

    # the first RNN layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),

    # the second/stacking RNN layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),

    # first dense layer
    tf.keras.layers.Dense(24, activation='relu'),

    # single-neuron dense layer for classification
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
# the model shows good 30-epoch results: train_acc=73%, val_acc=72%, val_loss=0.55 (also early stopping is good)

# test some sentences
sample_sentences = ["It Was, For, Uh, Medical Reasons, Says Doctor To Boris Johnson, Explaining Why They Had To Give Him Haircut",
                  "It's a beautiful sunny day",
                  "I lived in Ireland, so in High School they made me learn to speak and write in Gaelic",
                  "Census Foot Soldiers Swarm Neighborhoods, Kick Down Doors To Tally Household Sizes"]

sample_sequences=tokenizer.texts_to_sequences(sample_sentences)
print(sample_sequences)

padded_sample=pad_sequences(sample_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(padded_sample)
# result: close to 0.5=neutral; close to 1=sarcastic; close to 0=nonsarcastic
print(model.predict(padded_sample))
