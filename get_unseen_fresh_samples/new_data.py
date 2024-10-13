import pandas as pd
from os import listdir
import os
df = pd.DataFrame()
partial = 0
for f in listdir("csvs_a_pezzi_unix/"):
    temp_df = pd.read_csv(f"csvs_a_pezzi_unix/{f}", sep='\t',encoding='latin')
    temp_df = temp_df[temp_df['species_name'] == " "]
    partial += temp_df.shape[0]
    df = pd.concat([df,temp_df])

df.to_csv("df_unknown_species.csv", index=False, sep='\t', encoding='latin')
print(df.shape)