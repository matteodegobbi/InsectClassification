import urllib.request 
import pandas as pd
import time
import os.path

df = pd.read_csv('only_five_missing.csv')
genus_list = sorted(set(df['genus_name'].to_list()))
genus_list = genus_list[:2]
print(genus_list)

str_genus = ""

for index, genus in enumerate(genus_list):
    
    str_genus += genus

    if len(str_genus) > 6000:

        url = f"http://v3.boldsystems.org/index.php/API_Public/specimen?taxon={str_genus}&format=tsv"
         
        fname = f"new_csvs_a_pezzi/{index}.tsv"
        
        str_genus = ""

        if not os.path.isfile(fname):
            try:
                urllib.request.urlretrieve(url, fname)
            except Exception as e:
                print(f"Error: {e}")
                break
        
        time.sleep(1)
    
    else:

        str_genus += '|'

if str_genus:
    url = f"http://v3.boldsystems.org/index.php/API_Public/specimen?taxon={str_genus}&format=tsv"
    fname = f"new_csvs_a_pezzi/{index}.tsv"
    if not os.path.isfile(fname):
        try:
            urllib.request.urlretrieve(url, fname)
        except Exception as e:
            print(f"Error: {e}")