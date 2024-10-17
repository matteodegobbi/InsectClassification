import pandas as pd
import requests
from time import sleep
from pathlib import Path
import os.path

output_folder = 'image_dataset/'

def save_image_from_url(url, output_folder):
    if os.path.isfile(os.path.join(output_folder, url[url.rfind('/')+1:])):
       return

    image = requests.get(url)
    output_path = os.path.join(
        output_folder, url[url.rfind('/')+1:]
    )
    with open(output_path, "wb") as f:
        f.write(image.content)
        sleep(1)


df = pd.read_csv('raw_data.csv', delimiter='\t', encoding='latin')
df = df.dropna(subset=['image_urls'])


urls = {}
for i,row in df.iterrows():
    if row['species_name'] in urls.keys():
        urls[row['species_name']].append(row['image_urls'].split('|')[0])
    else:
        urls[row['species_name']] = [row['image_urls'].split('|')[0]]

# for species in urls.keys():
#     if (species != ' '):
#         print(species, len(urls[species]))
#         sleep(1)

missing_urls = []
for species in urls.keys():
    print(species)
    if (species != ' '):
        urls_list = urls[species]
        while urls_list:
            url = urls_list.pop()
            try:
                Path(f'{output_folder}{species}').mkdir(parents=True, exist_ok=True)
                save_image_from_url(url,f'{output_folder}{species}')
            except Exception as e:
                print("fail", e)
                missing_urls.append(url)