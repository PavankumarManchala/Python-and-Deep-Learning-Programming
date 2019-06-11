import requests
from bs4 import BeautifulSoup

wiki = "https://en.wikipedia.org/wiki/Deep_learning" \
       ""
page = requests.get(wiki).text
soup = BeautifulSoup(page, "html.parser")
anchors = soup.find_all("a")
title = soup.find("title")
href = soup.find("href")
print("---------------------Title------------------")
print(title.get_text())
for anchor in anchors:
    print("XXXXXXXXXXXAnchor tagXXXXXXXXXXX")
    print(anchor.get_text())
    print("|||||||||||||Href|||||||||||||||")
    print(anchor.get('href'))
