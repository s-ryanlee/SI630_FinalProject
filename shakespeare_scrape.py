import warnings
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_URL = 'https://github.com/tokestermw/tensorflow-shakespeare/tree/master/data/shakespeare/sparknotes/merged/'
data_path = "data/shakespeare/"
shake_html_path = Path(data_path+"shakespeare_response.html")

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

print('COLLECTING DATA')
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

print(f'Number of tables collected: {len(shakespeare_plays)}')

print('PROCESSING DATA')
shake_df = pd.concat(shakespeare_plays, axis=0)
shake_original = shake_df[shake_df['version'] == 'original'].copy().rename(columns={'text': 'original_text'})
shake_modern = shake_df[shake_df['version'] == 'modern'].copy().rename(columns={'text': 'modern_text'})
shake_aligned = shake_original.merge(
    shake_modern, how='left', on='play_line_id'
    ).drop(columns=['version_x', 'title_y', 'version_y', 'line_id_y']
    ).rename(columns={'title_x': 'title', 'line_id_x': 'line'})
shake_aligned = shake_aligned[['play_line_id', 'title', 'line', 'original_text', 'modern_text']]

print("CLEANING DATA")
shake_aligned['original_text_cleaned'] = shake_aligned.original_text.str.lower().str.strip()
shake_aligned['modern_text_cleaned'] = shake_aligned.modern_text.str.lower().str.strip()
# remove special characters
shake_aligned.replace(to_replace={
    'original_text_cleaned': r"[^A-Za-z\s0-9]",
    'modern_text_cleaned': r"[^A-Za-z\s0-9]",
}, value=" ", regex=True, inplace=True)
# fix any added whitespaces from first replace
shake_aligned.replace(to_replace={
    'original_text_cleaned': r"\s\s+",
    'modern_text_cleaned': r"\s\s+",
}, value=" ", regex=True, inplace=True)

shake_aligned['combined_text_clean'] = shake_aligned.modern_text_cleaned.str.strip() + " [SEP] " + shake_aligned.original_text_cleaned.str.strip()
print(f'DATA SAMPLE\n{shake_aligned.head()}')


print(f'CREATING TRAIN, DEV, TEST DATA SPLITS')
train, all_test = train_test_split(shake_aligned.index, train_size=0.8)
train_df = shake_aligned.iloc[train].copy().reset_index(drop=True)
all_test_df = shake_aligned.iloc[all_test].copy().reset_index(drop=True)
print(f'Number of training rows: {train_df.shape[0]}')
print(f'Number of dev+test rows: {all_test_df.shape[0]}')

dev, test = train_test_split(all_test_df.index, train_size=0.5)
dev_df = all_test_df.iloc[dev].copy().reset_index(drop=True)
test_df = all_test_df.iloc[test].copy().reset_index(drop=True)
print(f'Number of dev rows: {dev_df.shape[0]}')
print(f'Number of test rows: {test_df.shape[0]}')

print('WRITING DATA FILES')
shake_aligned.to_csv(data_path+"shakespeare_translated.csv", index=False)
train_df.to_csv(data_path+"train.csv", index=False)
dev_df.to_csv(data_path+"dev.csv", index=False)
test_df.to_csv(data_path+"test.csv", index=False)
