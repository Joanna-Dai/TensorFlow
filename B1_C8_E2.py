# creating text by using extended dataset

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
corpus = data.lower().split("\n")

