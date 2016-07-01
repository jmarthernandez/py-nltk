from nltk.corpus import wordnet
from pprint import pprint as pp

syns = wordnet.synsets('program');

# Syset
# pp(syns)

# Just the name of synomym
# pp(syns[0].lemmas()[0].name())

# Definition
# pp(syns[0].definition())

# Examples
# pp(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets('good'):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

pp(set(synonyms))
pp(set(antonyms))


# Semantic similarity
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')

pp(w1.wup_similarity(w2))
