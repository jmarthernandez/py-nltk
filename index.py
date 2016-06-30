from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from utils import read

mlk = read('mlk.txt')

print(sent_tokenize(mlk))
print(word_tokenize(mlk))
