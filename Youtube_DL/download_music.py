# 출처: http://www.youtube.com/watch?v=BaW_jenozKc

from __future__ import unicode_literals
import youtube_dl
from bs4 import BeautifulSoup
import urllib
import requests
import os
import numpy as np
import librosa.display
import librosa
from pydub import AudioSegment
import matplotlib.pyplot as plt

query_page = 'http://www.youtube.com/results?search_query='
mv_page = 'https://www.youtube.com'

ydl_opts = {
    'format': 'bestaudio/best',''
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}


def get_youtube_link(query):
    query = query.lower()                               # query를 소문자로 변형
    path = urllib.parse.quote_plus(query)               # string을 url 형태로 변환

    page = requests.get(query_page + path)               # youtube 검색 url 생성후 페이지 요청
    soup = BeautifulSoup(page.content, 'html.parser')
    html_page = soup.prettify()

    idx = html_page.lower().find('title="' + query)     # 먼저 쿼리와 관련된 제목을 찾는다.
    ridx = html_page.rfind("href=", 0, idx)             # 쿼리로 부터 'href='를 가진 문자의 위치를 거꾸로 찾는다

    if idx == -1:
        return ''

    page_str = ''
    ridx += 6
    while True:
        if html_page[ridx] == '"':
            break;
        page_str += html_page[ridx]
        ridx = ridx + 1

    return mv_page + page_str


def search_youtube_link(query):
    query = query.lower()                               # query를 소문자로 변형
    path = urllib.parse.quote_plus(query)               # string을 url 형태로 변환

    page = requests.get(query_page + path)               # youtube 검색 url 생성후 페이지 요청
    soup = BeautifulSoup(page.content, 'html.parser')
    html_page = soup.prettify()

    idx = html_page.lower().find('<h3 class="yt-lockup-title')     # 먼저 youtube 검색시 제목 표시란을 찾는다.
    new_idx = html_page.find("href=", idx)             # 쿼리로 부터 'href='를 가진 문자의 위치를 거꾸로 찾는다

    if idx == -1:
        return ''

    page_str = ''
    new_idx += 6
    while True:
        if html_page[new_idx] == '"':
            break;
        page_str += html_page[new_idx]
        new_idx += 1

    return mv_page + page_str


def extract_feature(file_name, save_file_name):
    y, sr = librosa.load(file_name)

    # Use a pre-computed power spectrogram with a larger frame
    s = np.abs(librosa.stft(y, n_fft=4096))**2
    chroma = librosa.feature.chroma_stft(S=s, sr=sr)    # S: power spectrogram

    # array([[ 0.685,  0.477, ...,  0.961,  0.986],
    # [ 0.674,  0.452, ...,  0.952,  0.926],
    # ...,
    # [ 0.844,  0.575, ...,  0.934,  0.869],
    # [ 0.793,  0.663, ...,  0.964,  0.972]])

    plt.subplot(1, 1, 1)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma)
    plt.tight_layout()
    plt.savefig(save_file_name, bbox_inches='tight', pad_inches=0)


def segmentation_mp3(load_file, save_file):
    sound = AudioSegment.from_mp3(load_file)  # for happy_song.mp3
    duration = 60 * 1000
    sound[:duration].export(save_file + '-begin.mp3', format="mp3")
    sound[duration:duration * 2].export(save_file + '-middle.mp3', format="mp3")
    sound[-duration:].export(save_file + '-end.mp3', format="mp3")


# 다운로드할 폴더명, 분할된 음악을 저장할 폴더, 음악 이미지를 저장할 폴더, 다운로드할 가수 및 노래 제목이 있는 폴더
def download_youtube_mp3(download_dir, split_music_dir, music_image_dir, file):

    ydl_opts['outtmpl'] = r'D:\MyPythonProject\Youtube_DL\download_file\happy_music\%(title)s.%(ext)s'

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        f = open(file)
        file_list = []
        idx = 0
        while True:
            line = f.readline()
            if not line:
                break

            line = line[:len(line) - 1]  # 맨뒤의 개행문자 삭제
            items = line.split('/')

            music_name = items[1] + ' - ' + items[0]
            link = search_youtube_link(music_name)
            file_list.append(link)
            file_name = music_name + '.mp3'

            if link == '':
                continue

            ydl.download(file_list[idx:])
            idx += 1

            save_file_name = items[1] + ' - ' + items[0]
            print(save_file_name)
            segmentation_mp3(os.path.join(download_dir, file_name),
                             os.path.join(split_music_dir, save_file_name))

            extract_feature(os.path.join(split_music_dir, save_file_name + "-begin.mp3"),
                            os.path.join(music_image_dir, save_file_name + "-begin.jpg"))
            extract_feature(os.path.join(split_music_dir, save_file_name + "-middle.mp3"),
                            os.path.join(music_image_dir, save_file_name + "-middle.jpg"))
            extract_feature(os.path.join(split_music_dir, save_file_name + "-end.mp3"),
                            os.path.join(music_image_dir, save_file_name + "-end.jpg"))

        f.close()
        # print(file_list)
        # ydl.download(filelist)

"""
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    filename = 'youtube.txt'
    f = open(filename, 'r')

    filelist = []
    while True:
        line = f.readline()
        if not line: break
        filelist.append(line)
    f.close()
    ydl.download(filelist)
    #ydl.download(['https://www.youtube.com/watch?v=JGwWNGJdvx8'])
"""

"""
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    f = open('happy.txt')
    filelist = []
    while True:
        line = f.readline()
        line = line[:len(line) - 1] # 맨뒤의 개행문자 삭제
        print(line)
        if not line: break
        items = line.split('/')
        print(items[1] + ' - ' + items[0])
        link = get_youtube_link(items[1] + ' - ' + items[0])
        if link == '':
            continue
        filelist.append(link)
    print(filelist)
    #ydl.download(filelist)
"""

download_youtube_mp3(r'D:\MyPythonProject\Youtube_DL\download_file\happy_music',
                     r'D:\MyPythonProject\Youtube_DL\download_file\happy_split_music',
                     r'D:\MyPythonProject\Youtube_DL\download_file\happy_music_image', 'happy.txt')
