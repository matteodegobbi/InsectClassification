from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
from torchvision import transforms
import pandas as pd
import numpy as np 
import dataset_utils
import random

def extract_features(model : nn.Module, device :str ,save_to_disk : bool = False, save_name_prefix : str = ""):
    df = pd.read_csv('final_dataset.csv',index_col=0)
    imsize = 64
    tform = transforms.Compose([transforms.Resize((imsize,imsize)),
                                transforms.PILToTensor(),
                                transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize(0.5,0.5)])
    image_dataset = torchvision.datasets.ImageFolder("image_dataset/",transform=tform)

    ###DIVIDE DATASET INTO SPLITS### TODO: MOVE INTO ANOTHER FILE AS A FUNCTION IMPORTANT shuffle=False
    img2dna = dataset_utils.get_imgs_bold_id(image_dataset,df)

    nucleotides = df[['nucleotide','species_name','genus_name','processid','image_urls']]
    colonna_dna = df.loc[:,"nucleotide"]
    nucleotides.loc[:,'nucleotide'] = colonna_dna.apply(dataset_utils.one_hot_encoding)
    random.seed(42)
    X_train_val, X_test, y_train_val, y_test = dataset_utils.data_split(nucleotides,0.2,random_state=42)
    train_data = X_train_val
    train_data['species_name'] = y_train_val
    X_train, X_validation, y_train, y_validation = dataset_utils.data_split(train_data,0.2,drop_labels=False,random_state=42)
    train_indices, val_indices, test_indices = dataset_utils.image_splits_from_df(X_train,X_validation,X_test,image_dataset)
     
    class WholeDataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.targets = data.targets            
        def __getitem__(self, index):
            x = self.data[index][0]
            y = self.targets[index]
            
            return x, y
        
        def __len__(self):
            return len(self.data)
            
    whole_dataset = WholeDataset(image_dataset)
    n_classes = np.unique(whole_dataset.targets).shape[0]
    batch_size =  16
    train_imgs = torch.utils.data.Subset(whole_dataset, train_indices)
    val_imgs = torch.utils.data.Subset(whole_dataset, val_indices)
    test_imgs = torch.utils.data.Subset(whole_dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(train_imgs, batch_size=batch_size,shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_imgs, batch_size=batch_size,shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_imgs, batch_size=batch_size,shuffle=False, num_workers=2)
    dataloaders = {"train":train_loader,"val":val_loader,"test":test_loader}
    dataset_sizes = {'train': len(train_imgs.indices), 'val':len(val_imgs.indices),'test':len(test_imgs.indices)}

    ###ACTUAL EXTRACTION OF THE FEATURES###
    torch.cuda.empty_cache()
    model.eval()
    train_features = []
    train_labels = np.array([]) 
    with torch.no_grad():
        for batch, targets in dataloaders['train']:
            disc_dict = model.extract_features(batch.to(device),targets.to(device)) 
            features_torch = disc_dict['feature']
            features_targets_torch = targets
            train_labels = np.concatenate((train_labels, features_targets_torch.cpu().numpy()))
            train_features.append(features_torch.cpu().numpy())
            torch.cuda.empty_cache()

    train_features = np.concatenate(train_features)

    if save_to_disk:
        torch.save(torch.tensor(train_features),save_name_prefix+'img_train_features.pt')
        torch.save(torch.tensor(train_labels),save_name_prefix+'img_train_labels')
    torch.cuda.empty_cache()


    model.eval()
    val_features = []
    val_labels = np.array([])
    with torch.no_grad():
        for batch,targets in dataloaders['val']:
            disc_dict = model.extract_features(batch.to(device),targets.to(device)) 
            features_torch = disc_dict['feature']
            features_targets_torch = targets
            val_labels = np.concatenate((val_labels, features_targets_torch.cpu().numpy()))
            val_features.append(features_torch.cpu().numpy())
            torch.cuda.empty_cache()

    val_features = np.concatenate(val_features)


    if save_to_disk:
        torch.save(torch.tensor(train_features),save_name_prefix+'img_val_features.pt')
        torch.save(torch.tensor(train_labels),save_name_prefix+'img_val_labels')
    torch.cuda.empty_cache()


    model.eval()
    test_features = []
    test_labels = np.array([])
    with torch.no_grad():
        for batch,targets in dataloaders['test']:
            disc_dict = model.extract_features(batch.to(device),targets.to(device)) 
            features_torch = disc_dict['feature']
            features_targets_torch = targets
            test_labels = np.concatenate((test_labels, features_targets_torch.cpu().numpy()))
            test_features.append(features_torch.cpu().numpy())
            torch.cuda.empty_cache()

    test_features = np.concatenate(test_features)


    if save_to_disk:
        torch.save(torch.tensor(train_features),save_name_prefix+'img_test_features.pt')
        torch.save(torch.tensor(train_labels),save_name_prefix+'img_test_labels')

    return (train_features,train_labels),(val_features,val_labels), (test_features,test_labels)



