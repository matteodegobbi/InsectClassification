import urllib.request 
import pandas as pd
import time
import os.path

df = pd.read_csv('only_five_missing.csv')
genus_list = sorted(set(df['genus_name'].to_list()))
# genus_list = genus_list[:2]
# print(genus_list)

for index, genus in enumerate(genus_list):

    url = f"http://v3.boldsystems.org/index.php/API_Public/specimen?taxon={genus}&format=tsv"
        
    fname = f"new_csvs_a_pezzi/{genus}.tsv"

    if not os.path.isfile(fname):
        try:
            urllib.request.urlretrieve(url, fname)
        except Exception as e:
            print(f"Error {genus}: {e}")
    
    time.sleep(1)