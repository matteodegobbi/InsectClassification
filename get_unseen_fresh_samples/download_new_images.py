import pandas as pd
import os.path

df = pd.read_csv('unknown_species_new_samples.csv')

df = df.dropna(subset=['image_urls'])
#print(df['image_urls'])
for i,row in df.iterrows():
    #print(type(row['image_urls']))
    if isinstance(row['image_urls'],float):
        print(i)
    
urls = {}
print(df.shape)
for i,row in df.iterrows():
    if row['genus_name'] in urls.keys():
        urls[row['genus_name']].append(row['image_urls'].split('|')[0])
    else:
        urls[row['genus_name']] = [row['image_urls'].split('|')[0]]
print(len(urls))
print(urls)
missing_urls = []


output_folder = 'image_dataset/'
import requests
from time import sleep
from pathlib import Path

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

missing_urls = []
for genus in urls.keys():
    print(genus)
    urls_list = urls[genus]
    while urls_list:
        url = urls_list.pop()
        try:
            #decomment to use
            Path(f'{output_folder}{genus}').mkdir(parents=True, exist_ok=True)
            save_image_from_url(url,f'{output_folder}{genus}')
        except Exception as e:
            print("fail",e)
            missing_urls.append(url)