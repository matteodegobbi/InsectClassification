import torch
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
im = torchvision.datasets.ImageFolder("image_dataset/")
df = pd.read_csv('final_dataset.csv',index_col=0)

lista = []
for i in range(len(im.imgs)):
    try:
        img, label = im[i]
        species_name = (list(im.class_to_idx.keys())[label]).replace('_',' ')
    except:
        lista.append(im.imgs[i])

print(f"missing {len(lista)} corrupted images, starting download...")
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
urls_list = []
for i,row in df.iterrows():
    for url in row['image_urls'].split('|'):
        urls_list.append(url)


missing = []
for filename in lista:
    name = filename[0][filename[0].rfind('/')+1:]
    for url in urls_list:
        if name in url:
            missing.append(url)

import os
import requests
from time import sleep
for url in missing:
    save_image_from_url(url,'temp')
    sleep(1)
