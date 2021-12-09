#download csv dataset (twitter data on sentiment analysis) and save into local folder
import os
import requests

url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/binary-emotion.csv"
folder_dir = "C:/Users/user/tensorflow_datasets"
file_name = "binary-emotion.csv"
response=requests.get(url)
with open(os.path.join(folder_dir, file_name), 'wb') as f:
    f.write(response.content)


#define stopwords and punctuations
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


import csv

#full range of emotional labels as a string containing the text
sentences = []
#sentiment is negative(0) or positive(1)
labels = []
with open(file_name, encoding='UTF-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    #loop each row
    for row in reader:
        labels.append(int(row[0]))
        sentence = row[1].lower()
        # add spaces around the punctuation
        sentence = sentence.replace(",", " , ")
        sentence = sentence.replace(".", " . ")
        sentence = sentence.replace("-", " - ")
        sentence = sentence.replace("/", " / ")
        # use beautifulsoup to strip html content
        soup = BeautifulSoup(sentence)
        sentence = soup.get_text()
        words = sentence.split()
        filtered_sentence = ""
        for word in words:
            #remove punctuations
            word = word.translate(table)
            #remove stopwords
            if word not in stopwords:
                filtered_sentence = filtered_sentence + word + " "
        sentences.append(filtered_sentence)
#check length of two value-sets (result: 35,327)
print(len(labels))
print(len(sentences))
#check the content of the first cleaned row
print(labels[0])
print(sentences[0])


#split dataset into training and test subsets for training a model
training_size=28000

training_sentences=sentences[0:training_size]
testing_sentences=sentences[training_size:]
training_labels=labels[0:training_size]
testing_labels=labels[training_size:]

print(len(training_sentences))


#create the word index based on training set and use it for testing sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size=20000 #voc is up to 20000 words
max_length=10 #each sentence is up to 10 words
trunc_type='post' #trancate longer sentences by cutting off the end
padding_type='post' #pad shorter sentences at the end
oov_tok='<OOV>' #use <OOV> (Out Of Vocabulary) to replace the word that is previously unseen in training set

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
#create the word index based on training set
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(training_sequences[0])
print(training_padded[0])
print(word_index)
print(testing_sequences[77])
print(testing_padded[77])