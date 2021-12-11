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
vocab_size = 2000 #adjust from 10000 to 2000
max_length = 85 #adjust from 100 to 85
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


# exploring vocab_size
from collections import OrderedDict
wc = tokenizer.word_counts
# sort vocab into descending order of word volume
newlist = (OrderedDict(sorted(wc.items(), key=lambda t: t[1], reverse=True)))
# print(newlist)
# plot word frequency
# the 'hockey stick' curve shows very few words are used many times while most words are used very few times
# but every word is weighted equally (has an 'entry' in the embedding), with large training set and samll val set, that's why many words in train set aren't present in val set
import matplotlib.pyplot as plt
xs=[]
ys=[]
curr_x = 1
for item in newlist:
  xs.append(curr_x)
  curr_x=curr_x+1
  ys.append(newlist[item])

plt.plot(xs,ys)
# look at the volume of the words (no. 300-10000 on the x-axis) with the scale from 0-100 on the y-axis
plt.axis([300, 10000, 0, 100])
# result: we found for the words in positions 2000-10000, they are ued less than 20 times in entire corpus, which also explains the overfitting
# therefore, we changed the vocab_size from 10000 to 2000 (note cut based on 20 times frequency is arbitrary)
plt.show()


# explore max_length
# result: ~26090 (i.e. 26090/26709=98%) sentences have no-more-than 85 words, max_length=85 can improve model performance
xs=[]
ys=[]
current_item=1
for item in sentences:
    xs.append(current_item)
    current_item=current_item+1
    ys.append(len(item))
newys=sorted(ys)
plt.plot(xs, newys)
plt.show()


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
# best practice for embedding_dim is fourth root of vocab_size, also, lower embedding_dim means faster training speed
embedding_dim=7 # fourth root of 2000 is 6.69
# model architecture
model=tf.keras.Sequential([

    # initialize embedding layer: every word in 10000 vocab size will be assigned 16 dimensional vectors
    # note: vocab_size in tokenization stage and embedding stage should be aligned
    # backpropagation
    tf.keras.layers.Embedding(vocab_size, embedding_dim),

    # feed the output embedding layer into a dense layer by using pooling
    # GlobalAveragePoooling: the dimensions of the embeddings are averaged out to produce a fixed-length output vector
    # 0 trainable parameters
    tf.keras.layers.GlobalAveragePooling1D(),

    # 24-neuron dense layer: hyperparameter tunning to learn #para=24-neuron x (16-dimension +1-bias) = 24x17=408
    # reduce 24 to 8 (with embedding_dim=7)
    tf.keras.layers.Dense(8, activation='relu'),

    # dense layer with L2 regularization (ridge regression, amplify the differences between nonzero, zero close-to-zero values, more commonly used in NLP, note L1 called lasso, helps to ignore the zero or close-to-zero weights)
    #                                    take a floating-point value as the regularization factor, result: smooth out training loss and validation loss somewhat (74% train acc, 71% val acc, 0.6 val loss, 30-epoch)
    # tf.keras.layers.Dense(8, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),

    # drop out 2 out of 8 neurons, but it doesn't work well as dropout is not good to reduce overfitting if there are very few neurons
    # tf.keras.layers.Dropout(.25)

    # single-neuron dense layer for binary classification: #para=1-neuron x 24-output of previous layer + 1-bias=25
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# change the default learning rate (0.001) for adam to mitigate overfitting (likey caused by network learns too quickly)
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
# optimizer='adam' is using default adam
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

#check num of para
model.summary()

# pre-adjustment: result after 30 epochs - training accuracy=98%, validation accuracy=76% (likely due to validation data contains words that aren't present in training data)
#                       indicators of overfitting: over the time, validation accuracy drops a little bit but validation loss increases sharply (val_loss=2.2)
#                       (i.e. the neural network is good at matching patterns in 'noisy' data in the training set that doesn't exist anywhere else. The better it gets at matching it, the worse the loss of the validation set will be.)
# lower learning_rate: 30-epoch result - training accuracy=92%, validation accuracy=80%, val_loss=0.51 (smaller val loss)
# + lower vocab_size: 30-epoch result - training accuracy=82%, validation accuracy=76%, val_loss=0.51 (closer train & val accuracy)
# + lower embedding_dim: 30-epoch result - training accuracy=81%, validation accuracy=76%, val_loss=0.51 (closer accuracy & faster speed)
# + less neurons for dense layer: 30-epoch result - similar accuracy & loss, the lines (of train, val) being less jaggy
# + with 100 epochs, train acc=0.82%, val acc=0.76%, val loss=0.54
num_epochs = 100 # adjusted from 30 to 100
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)


# use model to classify a sentence
sample_sentences=["granny starting to fear spiders in the garden might be real",
                  "game of thrones season finale showing this sunday night",
                  "TensorFlow book will be a best seller"]

sample_sequences=tokenizer.texts_to_sequences(sample_sentences)
print(sample_sequences)

padded_sample=pad_sequences(sample_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(padded_sample)
# result: higher values indicate likely sarcasm (the first sentence shows higher score)
print(model.predict(padded_sample))


# visualize embeddings
reverse_word_index = dict([value, key] for (key, value) in word_index.items())
e = model.layers[0] # embeddings layer
weights = e.get_weights()[0] #extract the weights of the vectors in the embeddings
print(weights.shape) # (2000,7) given 2000 vocab_size and 7 embedding_dim
# explore a word and its vector details
print(word_index['<OOV>']) # word_index start from 1
print(reverse_word_index[2]) # the most frequently used word apart from <OOV>
print(weights[2]) # the word is represented with 7 coefficients on the 7 dimensions/axes
# embedding projector uses two tab-seperated values (tsv) fils as full details, 1 for vector dimensions and 1 for metadata (narratives)
import io
out_v = io.open('C:/Users/user/tensorflow_datasets/vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('C:/Users/user/tensorflow_datasets/meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word+"\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
# visualize embeddings by loading above vector & meta tsv files over http://projector.tensorflow.org/