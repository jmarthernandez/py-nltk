from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from pprint import pprint as pp
from utils import read

mlk = read('mlk.txt')
stop_words = set(stopwords.words('english'))
words = word_tokenize(mlk)
filtered_mlk = []

for w in words:
    if w not in stop_words:
        filtered_mlk.append(w)

# Fancy One Liner
# filtered_mlk = [w in words if w not in stop_words]

pp(filtered_mlk)
