import pandas as pd

df = pd.read_csv('df_unknown_species.csv', delimiter='\t', encoding='latin')

df = df.dropna(subset=['image_urls']) #remove dnas with no corresponding image

# if there are more than 1000 samples pick only 1000
n_samples = 1000
sampled_df = df.groupby('genus_name').apply(lambda x: x.sample(frac=n_samples/len(x) if len(x) > n_samples else 1.0)).reset_index(drop=True)

sampled_df.to_csv('unknown_species_new_samples.csv', index=False)