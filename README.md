# py-nltk
NLP with NLTK

# Tokenizing
Tokenizing is basically a regex that pulls apart language based on what you need.
I use both word tokenizing and sentence tokenizing to pull apart the famous
"I Have a Dream" speech.

## Word Tokenizing
````
txt = "I am an example sentence.  Sometimes I have many sentences.  Look it's Mr. Justin"
word_tokenize(txt)
````

returns 

````
['I', 'am', 'an', 'example', 'sentence', '.', 'Sometimes', 'I', 'have', 'many', 'sentences', '.', 'Look', 'it', "'s", 'Mr.', 'Justin']
````

Word tokenizing treats punctuation as important because it is relevant to the meaning
of the text.
 
## Sentence Tokenizing
````
txt = "I am an example sentence.  Sometimes I have many sentences.  Look it's Mr. Justin"
sent_tokenize(txt)
````

returns 

````
['I am an example sentence.', 'Sometimes I have many sentences.', "Look it's Mr. Justin"]
````

One of the main benefits of NLTK is that the contributors have taken care of the frustrating
edge cases that we may not consider.  For example "Mr. Justin" is not considered two separate
sentences.

## Stop Words
````
txt = "I am an example sentence.  Sometimes I have many sentences.  Look it's Mr. Justin"
words_txt = word_tokenize(txt)
stop_words = set(stopwords.words('english'))
filtered_words = []

for w in words:
	if w not in stop_words:
		filtered_words.append(w)
````

returns 

````
['I', 'example', 'sentence', '.', 'Sometimes', 'I', 'many', 'sentences', '.', 'Look', "'s", 'Mr.', 'Justin']
````

Stop words are words that are so common that they generally don't change the meaning of a sentence.  Words like "the", "a", "and" do not have much impact on how a sentence is interpreted.