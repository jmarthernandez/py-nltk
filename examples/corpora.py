from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize
from pprint import pprint as pp

sample = gutenberg.raw('bible-kjv.txt')

sentences = sent_tokenize(sample)

pp(sentences[0:10])