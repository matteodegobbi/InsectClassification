from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
from torchvision import transforms
import pandas as pd
import numpy as np 
import dataset_utils
import random

def extract_image_features(model : nn.Module, device :str ,save_to_disk : bool = False, save_name_prefix : str = ""):
    
    all_images = torch.load('tensor_dataset/all_images.pt')
    all_dnas = torch.load('tensor_dataset/all_dnas.pt')
    all_labels = torch.load('tensor_dataset/all_labels.pt')
    train_loc = torch.load('tensor_dataset/train_loc.pt')
    val_seen_loc = torch.load('tensor_dataset/val_seen_loc.pt')
    val_unseen_loc = torch.load('tensor_dataset/val_unseen_loc.pt')
    test_seen_loc = torch.load('tensor_dataset/test_seen_loc.pt')
    test_unseen_loc = torch.load('tensor_dataset/test_unseen_loc.pt')
    species2genus = torch.load('tensor_dataset/species2genus.pt')
    described_labels_train = torch.load('tensor_dataset/described_species_labels_train.pt')
    described_labels_trainval= torch.load('tensor_dataset/described_species_labels_trainval.pt')
        
    class ImageDataset(Dataset):
            def __init__(self, imgs, targets):
                self.data = imgs
                self.targets = targets           
            def __getitem__(self, index):
                x = self.data[index]
                y = self.targets[index]
                return x, y
            
            def __len__(self):
                return len(self.data)
    dataset = ImageDataset(all_images,all_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,shuffle=False, num_workers=2)

    ###ACTUAL EXTRACTION OF THE FEATURES###
    torch.cuda.empty_cache()
    model.eval()
    features = []
    labels = np.array([]) 
    with torch.no_grad():
        for batch, targets in dataloader:
            disc_dict = model.extract_features(batch.to(device),targets.to(device)) 
            features_torch = disc_dict['feature']
            features_targets_torch = targets
            labels = np.concatenate((labels, features_targets_torch.cpu().numpy()))
            features.append(features_torch.cpu().numpy())
            torch.cuda.empty_cache()

    features = np.concatenate(features)
    train_features = features[train_loc]
    train_labels = labels[train_loc]
    
    val_features = features[torch.cat((val_seen_loc,val_unseen_loc))]
    val_labels = labels[torch.cat((val_seen_loc,val_unseen_loc))]
    test_features = features[torch.cat((test_seen_loc,test_unseen_loc))]
    test_labels = labels[torch.cat((test_seen_loc,test_unseen_loc))]
    if save_to_disk:
        torch.save(torch.tensor(train_features),save_name_prefix+'img_train_features.pt')
        torch.save(torch.tensor(train_labels),save_name_prefix+'img_train_labels.pt')
    torch.cuda.empty_cache()
    return (train_features,train_labels),(val_features,val_labels), (test_features,test_labels)



def extract_expanded_dna_features(model : nn.Module,device :str ,save_to_disk : bool = False, save_name_prefix : str = ""):
    ##loading  dataset and DIVIDING INTO SPLITS, TODO: move into another function
    df = pd.read_csv('final_dataset.csv',index_col=0)
    tform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.PILToTensor(),
                                transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize(0.5,0.5)])
    image_dataset = torchvision.datasets.ImageFolder("image_dataset/",transform=tform)
    species2genus = dataset_utils.species_label_to_genus_label(df,image_dataset)
    batch_size = 1000 

    img2dna = dataset_utils.get_imgs_bold_id(image_dataset,df)
    
    nucleotides = df[['nucleotide','species_name','genus_name','processid','image_urls']]
    colonna_dna = df.loc[:,"nucleotide"]
    nucleotides.loc[:,'nucleotide'] = colonna_dna.apply(dataset_utils.one_hot_encoding)
    random.seed(42)
    
    X_train_val, X_test, y_train_val, y_test = dataset_utils.data_split(nucleotides,0.2,random_state=42)
    print(y_test)
    train_data = X_train_val
    train_data['species_name'] = y_train_val
    
    X_train, X_validation, y_train, y_validation = dataset_utils.data_split(train_data,0.2,drop_labels=False,random_state=42)
    train_indices, val_indices, test_indices = dataset_utils.image_splits_from_df(X_train,X_validation,X_test,image_dataset)

    y_train = y_train.apply(lambda x: image_dataset.class_to_idx[x.replace(' ','_')])
    y_test = y_test.apply(lambda x: image_dataset.class_to_idx[x.replace(' ','_')])
    y_validation= y_validation.apply(lambda x: image_dataset.class_to_idx[x.replace(' ','_')])
    y_train_val = y_train_val.apply(lambda x: image_dataset.class_to_idx[x.replace(' ','_')])
    class DNAdataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = data
            self.targets = torch.tensor(targets)
            #self.transform = transform
            
        def __getitem__(self, index):
            x = torch.tensor(np.float32(self.data[index][0])).unsqueeze(0)
            y = self.targets[index]
            
            #if self.transform:
            #    x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            #    x = self.transform(x)
            
            return x, y
        
        def __len__(self):
            return len(self.data)
    d_train = DNAdataset(X_train.values, y_train.values)
    d_val = DNAdataset(X_validation.values, y_validation.values)
    d_test = DNAdataset(X_test.values, y_test.values)
    dataloader_train = DataLoader(d_train, batch_size=32,shuffle=False)
    dataloader_val = DataLoader(d_val, batch_size=32,shuffle=False)
    dataloader_test = DataLoader(d_test, batch_size=32,shuffle=False)
    dataloaders = {'train':dataloader_train,'val':dataloader_val,'test':dataloader_test}
    dataset_sizes = {'train': d_train.data.shape[0], 'val':d_val.data.shape[0],'test':d_test.data.shape[0]}
    ###actual extraction of the feature from the model
    model.eval()
    with torch.no_grad():
        train_features = []
        train_dna_labels = np.array([]) 
        for dnas,labels in dataloaders['train']:
            dnas = dnas.to(device)
            features = model.feature_extract(dnas)
            train_dna_labels = np.concatenate((train_dna_labels, labels.cpu().numpy()))
            train_features.append(features.cpu().numpy())
            torch.cuda.empty_cache()
        train_dna_features = torch.tensor(np.concatenate(train_features))
        train_dna_labels = torch.tensor(train_dna_labels)
        
        val_features = []
        val_dna_labels = np.array([]) 
        for dnas,labels in dataloaders['val']:
            dnas = dnas.to(device)
            features = model.feature_extract(dnas)
            val_dna_labels = np.concatenate((val_dna_labels, labels.cpu().numpy()))
            val_features.append(features.cpu().numpy())
            torch.cuda.empty_cache()
        val_dna_features = torch.tensor(np.concatenate(val_features))
        val_dna_labels = torch.tensor(val_dna_labels)
        
        test_features = []
        test_dna_labels = np.array([]) 
        for dnas,labels in dataloaders['test']:
            dnas = dnas.to(device)
            features = model.feature_extract(dnas)
            test_dna_labels = np.concatenate((test_dna_labels, labels.cpu().numpy()))
            test_features.append(features.cpu().numpy())
            torch.cuda.empty_cache()
        test_dna_features = torch.tensor(np.concatenate(test_features))
        test_dna_labels = torch.tensor(test_dna_labels)

        '''train_dna_features = train_dna_features.cpu()
        val_dna_features = val_dna_features.cpu()
        test_dna_features = test_dna_features.cpu()'''
        
        #####get indices in imgs corresponding to dna indices

        img2dna_indices = dict()
        for k,v in img2dna.items():
            #print(k)
            #print(v)
            
            dna_index = np.where(X_train['processid'].values == v)#IF ITS IN TRAIN
            if not (dna_index[0].size > 0):
                dna_index = np.where(X_validation['processid'].values == v)#IF ITS IN VAL
                if not (dna_index[0].size > 0):
                     dna_index = np.where(X_test['processid'].values == v)#IF ITS IN TEST
        
            dna_index = dna_index[0][0]
            for i,(name,_) in enumerate(image_dataset.imgs):
                if name == k:
                    image_index = i
                    break
            img2dna_indices[image_index] = dna_index

        ####expanding features because for each dna we have more than 1 image
        dna_features2 = []
        dna_labels2 = []
        for i in train_indices:
            dna_features2.append(train_dna_features[img2dna_indices[i]])
            dna_labels2.append(train_dna_labels[img2dna_indices[i]])
        expanded_train_dna_features = torch.stack(dna_features2)
        expanded_train_dna_labels = torch.stack(dna_labels2)
        
        dna_features2 = []
        dna_labels2 = []
        for i in val_indices:
            dna_features2.append(val_dna_features[img2dna_indices[i]])
            dna_labels2.append(val_dna_labels[img2dna_indices[i]])
        expanded_val_dna_features = torch.stack(dna_features2)
        expanded_val_dna_labels = torch.stack(dna_labels2)

        dna_features2 = []
        dna_labels2 = []
        for i in test_indices:
            dna_features2.append(test_dna_features[img2dna_indices[i]])
            dna_labels2.append(test_dna_labels[img2dna_indices[i]])
        expanded_test_dna_features = torch.stack(dna_features2)
        expanded_test_dna_labels = torch.stack(dna_labels2)


        if save_to_disk:
            torch.save(torch.tensor(expanded_train_dna_features),save_name_prefix+'dna_train_features.pt')
            torch.save(torch.tensor(expanded_train_dna_labels),save_name_prefix+'dna_train_labels.pt')
            torch.save(torch.tensor(expanded_val_dna_features),save_name_prefix+'dna_val_features.pt')
            torch.save(torch.tensor(expanded_val_dna_labels),save_name_prefix+'dna_val_labels.pt')
            torch.save(torch.tensor(expanded_test_dna_features),save_name_prefix+'dna_test_features.pt')
            torch.save(torch.tensor(expanded_test_dna_labels),save_name_prefix+'dna_test_labels.pt')
        
    return (expanded_train_dna_features,expanded_train_dna_labels),(expanded_val_dna_features,expanded_val_dna_labels), (expanded_test_dna_features,expanded_test_dna_labels)