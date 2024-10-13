import pandas as pd

df = pd.read_csv('unknown_species_new_samples.csv')

genus = sorted(set(df['genus_name'].to_list()))

print(len(genus))
print(df.shape)
print(genus)