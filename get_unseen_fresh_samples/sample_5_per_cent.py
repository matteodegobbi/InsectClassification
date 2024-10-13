import pandas as pd

df = pd.read_csv('df_unknown_species.csv', delimiter='\t', encoding='latin')

df = df.dropna(subset=['image_urls'])

sample_fraction = 0.10

sampled_df = df.groupby('genus_name').apply(lambda x: x.sample(frac=sample_fraction if len(x) > 1 else len(x))).reset_index(drop=True)

sampled_df.to_csv('unknown_species_new_samples.csv', index=False)

