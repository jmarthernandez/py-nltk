# py-nltk
NLP with NLTK

# Tokenizing
Tokenizing is basically a regex that pulls apart language based on what you need.
I use both word tokenizing and sentence tokenizing to pull apart the famous
"I Have a Dream" speech.

## Word Tokenizing
````
from nltk.tokenize import word_tokenize

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
from nltk.tokenize import sent_tokenize

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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

## Coporas

Copora is a group of texts that have something similar.  Medical journals, State of the Union addresses, and Tweets would all be examples of Corporas.

## Lexicon
This is meaning of words within a context.  For example, bull would have two meanings if you asked an investor and a rancher.

## Tagging Parts of Speech

In NLTK it is incredibly simple to tag parts of speech using `nltk.pos_tag(word)`.

Here is an example where we iterate over a txt and print out the part of speech.

````
import nltk
from nltk.tokenize import word_tokenize

txt = "I am an example sentence.  Sometimes I have many sentences.  Look it's Mr. Justin"
words_txt = word_tokenize(txt)
tagged = nltk.pos_tag(words_text)
print(tagged)

````
returns

````
[('I', 'PRP'), ('am', 'VBP'), ('an', 'DT'), ('example', 'NN'), ('sentence', 'NN'), ('.', '.'), ('Sometimes', 'RB'), ('I', 'PRP'), ('have', 'VBP'), ('many', 'JJ'), ('sentences', 'NNS'), ('.', '.'), ('Look', 'VB'), ('it', 'PRP'), ("'s", 'VBZ'), ('Mr.', 'NNP'), ('Justin', 'NNP')]

````

### Part of Speech Tags

| POS Tag | Description                           | Example                                 |
|---------|---------------------------------------|-----------------------------------------|
| CC      | coordinating conjunction              | and                                     |
| CD      | cardinal number                       | 1, third                                |
| DT      | determiner                            | the                                     |
| EX      | existential there                     | there is                                |
| FW      | foreign word                          | d’hoevre                                |
| IN      | preposition/subordinating conjunction | in, of, like                            |
| JJ      | adjective                             | big                                     |
| JJR     | adjective, comparative                | bigger                                  |
| JJS     | adjective, superlative                | biggest                                 |
| LS      | list marker                           | 1)                                      |
| MD      | modal                                 | could, will                             |
| NN      | noun, singular or mass                | door                                    |
| NNS     | noun plural                           | doors                                   |
| NNP     | proper noun, singular                 | John                                    |
| NNPS    | proper noun, plural                   | Vikings                                 |
| PDT     | predeterminer                         | both the boys                           |
| POS     | possessive ending                     | friend‘s                                |
| PRP     | personal pronoun                      | I, he, it                               |
| PRP$    | possessive pronoun                    | my, his                                 |
| RB      | adverb                                | however, usually, naturally, here, good |
| RBR     | adverb, comparative                   | better                                  |
| RBS     | adverb, superlative                   | best                                    |
| RP      | particle                              | give up                                 |
| TO      | to                                    | to go, to him                           |
| UH      | interjection                          | uhhuhhuhh                               |
| VB      | verb, base form                       | take                                    |
| VBD     | verb, past tense                      | took                                    |
| VBG     | verb, gerund/present participle       | taking                                  |
| VBN     | verb, past participle                 | taken                                   |
| VBP     | verb, sing. present, non-3d           | take                                    |
| VBZ     | verb, 3rd person sing. present        | takes                                   |
| WDT     | wh-determiner                         | which                                   |
| WP      | wh-pronoun                            | who, what                               |
| WP$     | possessive wh-pronoun                 | whose                                   |
| WRB     | wh-abverb                             | where, when                             |

## Chunking

Chunking is recovering phrase groups based on parts of speech tags.
We can parse out verb groups, the predicate, or noun phrases, the subject.
we can break apart sentences into its chunks.

## Named Entities

Named entities are a more specific way of identifying parts of text.
While they are much more readable than.

````
import nltk
from nltk.tokenize import word_tokenize

txt = "I am an example sentence.  Sometimes I have many sentences.  Look it's Mr. Justin"
words_txt = word_tokenize(txt)
tagged = nltk.pos_tag(words_text)
print(nltk.ne_chunk(tagged))

````
returns

````
Tree('S', [('I', 'PRP'), ('am', 'VBP'), ('an', 'DT'), Tree('GPE', [('American', 'JJ')]), ('Dollar', 'NN'), ('from', 'IN'), ('June', 'NNP'), ('of', 'IN'), Tree('ORGANIZATION', [('WHO', 'NNP'), ('Obama', 'NNP')])])

````

As we can see `ne_chunk` is not super accurate but could potentially provide insights.
For example we see that American is tagged as "JJ" rather than GPE.

| NE Type      | Examples                                |
|--------------|-----------------------------------------|
| ORGANIZATION | Georgia-Pacific Corp., WHO              |
| PERSON       | Eddy Bonte, President Obama             |
| LOCATION     | Murray River, Mount Everest             |
| DATE         | June, 2008-06-29                        |
| TIME         | two fifty a m, 1:30 p.m.                |
| MONEY        | 175 million Canadian Dollars, GBP 10.40 |
| PERCENT      | twenty pct, 18.75 %                     |
| FACILITY     | Washington Monument, Stonehenge         |
| GPE          | South East Asia, Midlothian             |