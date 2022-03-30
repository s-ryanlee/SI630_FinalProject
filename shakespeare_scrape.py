import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = 'https://github.com/tokestermw/tensorflow-shakespeare/tree/master/data/shakespeare/sparknotes/merged/'
shake_response = requests.get(BASE_URL)
with open("data/shakespeare_response.txt", "w", encoding='utf-8') as file:
    file.write(shake_response.text)

soup = BeautifulSoup(shake_response.content, "html.parser")
results = soup.find(id="repo-content-pjax-container")
a_class = "js-navigation-open Link--primary"
a_elements = results.find_all("a", class_=a_class)

shakespeare_plays = []
for a_tag in a_elements:
    file_name = a_tag.text
    file_name_parts = file_name.split("_")
    play_title = file_name_parts[0].replace("-", " ")
    play_version = file_name_parts[1].replace(".snt.aligned", "")

    df = pd.read_html(BASE_URL+file_name, flavor="bs4", encoding="utf-8")[0]
    df = df.dropna(axis=1).rename(columns={1: "text"})
    df['title'] = play_title
    df['version'] = play_version
    shakespeare_plays.append(df)

shakespeare_plays_combined = pd.concat(shakespeare_plays, axis=0)
shakespeare_plays_combined.to_csv("data/shakespeare_plays_combined.csv", index=False, encoding="utf-8")
