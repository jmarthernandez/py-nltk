import nltk
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk.corpus import stopwords, state_union
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

# pp(filtered_mlk)

train_text = state_union.raw('2005-GWBush.txt')
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

custom_tokenized = custom_sent_tokenizer.tokenize(mlk)

def process_content():
    try:
        for i in custom_tokenized[5:]:
            words = word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<.*>+}
                            }<VB.?|IN|DT|TO>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
    except Exception as e:
        print(str(e))

process_content()