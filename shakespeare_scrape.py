import warnings
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd

warnings.filterwarnings("ignore")

BASE_URL = 'https://github.com/tokestermw/tensorflow-shakespeare/tree/master/data/shakespeare/sparknotes/merged/'
shake_html_path = Path("data\shakespeare\shakespeare_response.html")

if shake_html_path.exists():
    print("USING CACHE")
    with open(shake_html_path, "r", encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), "html.parser")
else:
    print("REQUESTING URL")
    shake_response = requests.get(BASE_URL)
    with open(shake_html_path, "w", encoding='utf-8') as file:
        file.write(shake_response.text)
        soup = BeautifulSoup(shake_response.content, "html.parser")

results = soup.find(id="repo-content-pjax-container")
a_class = "js-navigation-open Link--primary"
a_elements = results.find_all("a", class_=a_class)

print('PROCESSING DATA')
shakespeare_plays = []
for a_tag in a_elements:
    link_text = a_tag.text
    link_parts = link_text.split("_")
    play_title = link_parts[0].replace("-", " ")
    play_version = link_parts[1].replace(".snt.aligned", "")

    df = pd.read_html(BASE_URL+link_text, flavor="bs4", encoding="utf-8")[0]
    df = df.dropna(axis=1).rename(columns={1: "text"})
    df['title'] = play_title
    df['version'] = play_version
    df['line_id'] = df.index + 1
    df['play_line_id'] = df.line_id.astype(str) + df.title.str.replace(' ', '')

    shakespeare_plays.append(df)

print(f'Number of tables collected and processed: {len(shakespeare_plays)}')

shake_df = pd.concat(shakespeare_plays, axis=0)
shake_original = shake_df[shake_df['version'] == 'original'].copy().rename(columns={'text': 'original_text'})
shake_modern = shake_df[shake_df['version'] == 'modern'].copy().rename(columns={'text': 'modern_text'})
shake_aligned = shake_original.merge(
    shake_modern, how='left', on='play_line_id'
    ).drop(columns=['version_x', 'title_y', 'version_y', 'line_id_y']
    ).rename(columns={'title_x': 'title', 'line_id_x': 'line'})
shake_aligned = shake_aligned[['play_line_id', 'title', 'line', 'original_text', 'modern_text']]
print(f'DATA SAMPLE\n{shake_aligned.head()}')

shake_aligned.to_csv("data/shakespeare_translated.csv", index=False, encoding="utf-8")
