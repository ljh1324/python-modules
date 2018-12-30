import requests
from bs4 import BeautifulSoup
import urllib

home_page = 'http://www.youtube.com/results?search_query='
mv_page = 'https://www.youtube.com'

def get_youtube_link(query):
    query = query.lower()                             # query를 소문자로
    path = urllib.parse.quote_plus(query)             # string을 url 형태로 변환
    print(path)
    page = requests.get(home_page + path)             # youtube 검색 url 생성후 페이지 요청
    soup = BeautifulSoup(page.content, 'html.parser')
    html_page = soup.prettify()
    print(html_page)
    idx = html_page.lower().find('title="' + query)     # 먼저 쿼리와 관련된 제목을 찾는다.
    ridx = html_page.rfind("href=", 0, idx)             # 쿼리로 부터 'href='를 가진 문자의 위치를 찾는다
    print('idx: ', idx)
    print('ridx: ', ridx)
    str = ''
    ridx = ridx + 6
    while True:
        if html_page[ridx] == '"':
            break;
        str += html_page[ridx]
        ridx = ridx + 1

    return (mv_page + str)

f = open('happy.txt')
filelist = []
while True:
    line = f.readline()
    line = line[:len(line) - 1] # 맨뒤의 개행문자 삭제
    print(line)
    if not line: break
    items = line.split('/')
    print(items[1] + ' - ' + items[0])
    #link = get_youtube_link(items[1] + ' - ' + items[0])
    link = ""
    filelist.append(link)
f.close()
