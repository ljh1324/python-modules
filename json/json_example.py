import json
from pprint import pprint

with open('happy_song.json') as data_file:
    data = json.load(data_file)

#pprint(data) #data는 json 전체를 dictionary 형태로 저장하고 있음

#-----여기까지 동일-----

print(data["tracks"]["track"][0]["name"]) #값 하나하나 접근하기

