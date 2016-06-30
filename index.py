from nltk.tokenize import sent_tokenize, word_tokenize

txt_file = open('mlk.txt', 'r')
mlk = txt_file.read()
txt_file.close()

print(sent_tokenize(mlk))
print(word_tokenize(mlk))
