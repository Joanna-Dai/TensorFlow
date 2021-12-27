# creating text by using extended dataset
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np


# download txt file that has 1700 lines of text gathered from a number of songs for experimentation
import os
import requests

url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt"
folder_dir = "C:/Users/user/tensorflow_datasets"
file_name = "irish-lyrics-eof.txt"
response=requests.get(url)
with open(os.path.join(folder_dir, file_name), 'wb') as f:
    f.write(response.content)


# load the txt data into corpus
data = open(os.path.join(folder_dir, file_name)).read()

# windowing the data: to extend the size of dataset without adding new songs: input_sequences # = (number_of_words - window_size) x window_size
#   instead of line by line, we create a "window" by treating all lines as one long, continuous text and move the window forward one word to get the next input sequences
sentences = []
alltext = []
# tune-able hyperparameter: smaller window size --> more data, but there will be fewer words (sequence becomes smaller) to give a label and may end up with nonsensical poetry
max_sequence_len = 6
# converted to lower case
corpus = data.lower()
alltext.append(corpus)
# split into an array of words
words = corpus.split(" ")
print(len(words))

# i is the index of the first word of the smaller sequence
#   the largest possible value for i = range_size (note max_sequence_len = size of the moving window/smaller sequence)
range_size = len(words)-max_sequence_len
for i in range(0, range_size):
	thissentence=""
    # loop through each word and make sentence of each word from the current index up to the current index plus max_sequence_len (window size) --> more data & slower speed
	for word in range(0, max_sequence_len-1):
		word = words[i+word]
		thissentence = thissentence + word
		thissentence = thissentence + " "
    # add each newly constructed sentences into sentences array
	sentences.append(thissentence)
# on above way, we have number of smaller sequences (for training): range_size x max_sequence_len = (num_words - window_size) x window_size
print(sentences)

oov_tok = "<OOV>"
# tune-able hyperparameter: vocab_size for training (hereby the next word will be across 2700 words)
vocab_size = 2700
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, split=" ", char_level=False)
tokenizer.fit_on_texts(alltext)
# 2691 unique words
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)


# split the sentence/line into multiple smaller sequences
input_sequences = []
# loop each line in the sentences group we created
for line in sentences:
    # tokenize the line and change list "[[ ]]" to "[ ]" by putting [0]
	token_list = tokenizer.texts_to_sequences([line])[0]
    # loop smaller sequences of the line and start from (word 0 + word 1)
	for i in range(1, len(token_list)):
        # start from first two words and end with full line, note for list, list[:n+1]-->[0,n+1)-->[0,n]
		n_gram_sequence = token_list[:i+1]
        # append batch of smaller sequences of each line to the input_sequences
		input_sequences.append(n_gram_sequence)


# pad all input_sequences into a regular shape
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# split smaller sequences into features (the sequence without last word) and label (the last word of the sequence)
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
# encode labels into a set of Ys (given output is a set of labels) that can be used to train (can eat memory very quickly)
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


# model hierarchy
model = Sequential()
# tune-able hyperparameter: change dimension from 8 to 16
model.add(Embedding(total_words, 16))
# improve model by using multiple stacked LSTMs
# tune-able hyperparameter: the number of LSTMs --> change from window size (max_sequence_len=6) to 32
model.add(Bidirectional(LSTM(32, return_sequences='True')))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(total_words, activation='softmax'))
# model guess approach
# tune-able hyperparameter: upper the learning rate on the adam optimizer
# note: if encounter direct copy from the sentences in corpus --> best to reduce learning rate and decrease the number of LTSMs
adam = Adam(lr=0.01)
# model accuracy is not the best measurement for poetry: subjective examination is required --> no hard-and-fast rule to follow to determine whether the model is good or bad
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# 1000-epoch result (pre-tune): model accuracy=64%, loss=1.51
# 100-epoch result (post-tune):
# note: result of different users can be different due to the random initialization of the neurons will impact the final scores
history = model.fit(xs, ys, epochs=100, verbose=1)
model.save("bidiirish2.h5")

# graph epoch against model accuracy
import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

# create next words for seed text
seed_text = "sweet jeremy saw dublin"
next_words = 10
# repeat the prediction to have 10 next words (i.e. AI-created string of text)
for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen = max_sequence_len -1, padding='pre')
	# model.predict_classes will return category not value (i.e. it's de factor same with np.argmax(model.predict())
	# model.predict_classes were removed in tensorflow 2.6
	predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
	output_word = ""

	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word # append output word to generate the extended text after the seed text

print(seed_text)