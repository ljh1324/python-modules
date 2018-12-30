import nltk
sentence = """At eight o'clock on Thursday morning
... Arthur didn't feel very good."""

tokens = nltk.word_tokenize(sentence)
print(tokens)

tagged = nltk.pos_tag(tokens)
print(tagged[0:6])

f = open('test.txt', 'r', encoding='utf-8')
sentence = f.read()
f.close()

tokens = nltk.word_tokenize(sentence)
print(tokens)

tagged = nltk.pos_tag(tokens)
print(tagged[0:6])

f = open('강아지행동.txt', 'r')
sentence = f.read()
f.close()

tokens = nltk.word_tokenize(sentence)
print(tokens)

tagged = nltk.pos_tag(tokens)
print(tagged[0:6])