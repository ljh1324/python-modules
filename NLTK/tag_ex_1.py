from konlpy.tag import Hannanum

hannanum = Hannanum()

f = open('강아지행동.txt', 'r')
sentence = f.read()
f.close()

print(hannanum.analyze(sentence))
print(hannanum.morphs(sentence))
print(hannanum.nouns(sentence))
print(hannanum.pos(sentence))

from konlpy.tag import Kkma

kkma = Kkma()

print(kkma.morphs(sentence))
print(kkma.nouns(sentence))
print(kkma.pos(sentence))
print(kkma.sentences(sentence))

