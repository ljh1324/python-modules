import json
import requests
import urllib.request
from bs4 import BeautifulSoup

tag_api = 'http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag=%(tag)s&api_key=%(api_key)s&format=json&limit=%(limit)d'
api_key = 'fefa71dba98d98568c4cddc2f874ede4'


def jason_to_txt_file(tag, save_filename):
    with open(save_filename, 'w') as save_file:
        link = tag_api % {"api_key": api_key, "tag": tag, "limit": 500}
        print(link)
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        data = soup.prettify()
        data = json.loads(data)

        data_len = len(data["tracks"]["track"])
        for i in range(data_len):
            track = data["tracks"]["track"][i]["name"]
            artist = data["tracks"]["track"][i]["artist"]["name"]
            rank = str(data["tracks"]["track"][i]["@attr"]["rank"])

            save_line = track + "/" + artist + "/" + rank + '\n'
            print(save_line)
            save_file.write(save_line)

jason_to_txt_file('happy', 'happy.txt')