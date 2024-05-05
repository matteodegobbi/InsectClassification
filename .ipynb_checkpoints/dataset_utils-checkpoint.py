import numpy as np
import pandas as pd 
from typing import List
from sklearn.model_selection import train_test_split
import random 
import torch
from torchvision.utils import save_image
import os 
def one_hot_encoding(nucleotide: str, seq_len=658) -> np.ndarray:
    # Cutting the sequence if it is longer than a pre-defined value seq_len
    if len(nucleotide) > seq_len:
        nucleotide = nucleotide[:seq_len]
    # Encoding
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    sequence = [mapping[i] if i in mapping else 4 for i in nucleotide]
    encoded_sequence = np.eye(5)[sequence]
    # Padding if the sequence is smaller than a pre-defined value seq_len
    if len(encoded_sequence) < seq_len:
        padding = np.zeros((seq_len - len(encoded_sequence), 5))
        encoded_sequence = np.concatenate((encoded_sequence, padding))
    return encoded_sequence


def get_imgs_bold_id(image_dataset,df):
  img2dna = dict()
  not_found_images = []
  for i, row in df.iterrows():
      urls = row['image_urls'].split('|')
      species_name = row['species_name'].replace(' ','_')
      #if len(urls) >1 and row['species_name'] == 'Drosophila affinis':
      for url in urls:
          image_name_csv ='image_dataset/'+species_name+'/'+url[url.rfind('/')+1:]
          #print(image_name_csv)
          trovato = False
          for img in image_dataset.imgs:
              if img[0] == image_name_csv:
                  img2dna[img[0]]= row['processid']
                  trovato = True
                  break
          if not trovato:
              not_found_images.append(image_name_csv)
  return img2dna

def data_split(df, test_ratio,drop_labels = False):
    test = []
    genus_count = df.groupby('genus_name')['species_name'].nunique()
    
    for genus_name in genus_count.index:
        number_undescribed_species = genus_count[genus_name]//3
        species = list(df.loc[df['genus_name']==genus_name]['species_name'].unique())
        undescribed_species = random.sample(species,number_undescribed_species)
        test = test+undescribed_species

    df_remaining = df.loc[~df.species_name.isin(test)]
    df_undescribed = df.loc[df.species_name.isin(test)]
    
    y = df_remaining['species_name']
    if drop_labels:
        X = df_remaining.drop(columns=['species_name','genus_name','processid','image_urls'])
        X_undescribed = df_undescribed.drop(columns=['species_name','genus_name','processid','image_urls'])
    else:
        X = df_remaining
        X_undescribed = df_undescribed
    y_undescribed = df_undescribed['species_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    y_test = pd.concat([y_test,y_undescribed])
    X_test = pd.concat([X_test,X_undescribed])
    return X_train, X_test, y_train, y_test

def image_filenames_from_df(df: pd.core.frame.DataFrame) -> List[str]:
    filenames : List[str] = []
    for i,row in df.iterrows():
        urls = row['image_urls']
        for url in urls.split('|'):
            species_name = row['species_name'].replace(' ','_')
            image_filename = 'image_dataset/'+species_name+'/'+url[url.rfind('/')+1:]
            filenames.append(image_filename)
    return filenames

def image_splits_from_df(X_train, X_validation,X_test,image_dataset):
    train_filenames : List[str] = image_filenames_from_df(X_train)
    val_filenames : List[str] = image_filenames_from_df(X_validation)
    test_filenames : List[str] = image_filenames_from_df(X_test)
    train_indices : List[int] = []
    val_indices : List[int] = []
    test_indices : List[int] = []
    for i, (filename,label) in enumerate(image_dataset.imgs):
        if filename in train_filenames:
            train_indices.append(i)
        elif filename in val_filenames:
            val_indices.append(i)
        elif filename in test_filenames:
            test_indices.append(i)
        else:
            raise Exception(f"Exception: Filename {filename} isn't in any of the splits")
    return train_indices,val_indices,test_indices

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class Save_samples_params():
    def __init__(self,latent_tensors,sample_random_classes,sample_dir):
        self.latent_tensors = latent_tensors
        self.sample_random_classes = sample_random_classes
        self.sample_dir = sample_dir
class Fit_params():
    def __init__(self,discriminator_optimizer,generator_optimizer,discriminator,generator,dataloaders,device,writer,batch_size,n_classes,latent_size):
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.discriminator = discriminator
        self.generator = generator
        self.dataloaders = dataloaders
        self.device = device
        self.writer = writer
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.latent_size = latent_size
def save_samples(index, save_p : Save_samples_params, generator ,writer,show=True):
    with torch.no_grad():
        fake_images = generator(save_p.latent_tensors,save_p.sample_random_classes)
        fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
        save_image(denorm(fake_images), os.path.join(save_p.sample_dir, fake_fname), nrow=8)
        writer.add_image('sample image',denorm(fake_images[0]),global_step=index)
        print('Saving', fake_fname)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5