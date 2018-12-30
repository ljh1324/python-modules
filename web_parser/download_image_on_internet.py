import requests
from bs4 import BeautifulSoup
import urllib.request

image_page = 'http://www.youtube.com/results?search_query='
save_dir = 'myfire'


def download_image(site):
    page = requests.get(site)               # youtube 검색 url 생성
    soup = BeautifulSoup(page.content, 'html.parser')
    html_page = soup.prettify()
    start_idx = 0
    while True:
        idx = html_page.lower().find(".jpg", start_idx)     # 먼저 쿼리와 관련된 제목을 찾는다.
        if idx == -1:
            break
        ridx = html_page.rfind("//", 0, idx)             # 쿼리로 부터 'href='를 가진 문자의 위치를 찾는다
        link = html_page[ridx:idx + 4]
        item = link[link.rfind("/") + 1:]
        start_idx = idx + 4
        urllib.request.urlretrieve("https:" + link, save_dir + '/' + item)

my_link = 'https://pixabay.com/ko/photos/?min_height=&image_type=&cat=&q=fire&min_width=&order=popular&pagi='
download_image(my_link + '8')
download_image(my_link + '9')
download_image(my_link + '10')
download_image(my_link + '11')
download_image(my_link + '12')
download_image(my_link + '13')
download_image(my_link + '14')
download_image(my_link + '15')
