from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
from torchvision import transforms
import pandas as pd
import numpy as np 
import dataset_utils
import random
import scipy.io as io


def extract_image_features(model : nn.Module, device :str,args):
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
    
    
    # Read dataset
    matlab_dataset = io.loadmat(args.dataset_path)
    
    all_images = torch.tensor(matlab_dataset['all_images'])
    all_dnas = torch.tensor(matlab_dataset['all_dnas'])
    all_labels = torch.tensor(matlab_dataset['all_labels']).squeeze()-1
    train_loc = torch.tensor(matlab_dataset['train_loc']).squeeze()-1
    val_seen_loc = torch.tensor(matlab_dataset['val_seen_loc']).squeeze()-1
    val_unseen_loc = torch.tensor(matlab_dataset['val_unseen_loc']).squeeze()-1
    test_seen_loc = torch.tensor(matlab_dataset['test_seen_loc']).squeeze()-1
    test_unseen_loc = torch.tensor(matlab_dataset['test_unseen_loc']).squeeze()-1
    species2genus = torch.tensor(matlab_dataset['species2genus'])-1
    
    described_labels_train = matlab_dataset['described_species_labels_train'].squeeze()-1
    described_labels_trainval = matlab_dataset['described_species_labels_trainval'].squeeze()-1
    
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
    torch.cuda.empty_cache()
    return features,(train_features,train_labels),(val_features,val_labels), (test_features,test_labels)


def extract_expanded_dna_features(model : nn.Module,device :str,args):
   
    class DNAdataset(Dataset):
        def __init__(self, data, targets):
            self.data = data.float()
            self.targets = targets.clone().detach()
            
        def __getitem__(self, index):
            x = self.data[index].unsqueeze(0)
            y = self.targets[index]
            return x, y
        def __len__(self):
            return len(self.data)

    
    # Read dataset
    matlab_dataset = io.loadmat(args.dataset_path)
    
    all_images = torch.tensor(matlab_dataset['all_images'])
    all_dnas = torch.tensor(matlab_dataset['all_dnas'])
    all_labels = torch.tensor(matlab_dataset['all_labels']).squeeze()-1
    train_loc = torch.tensor(matlab_dataset['train_loc']).squeeze()-1
    val_seen_loc = torch.tensor(matlab_dataset['val_seen_loc']).squeeze()-1
    val_unseen_loc = torch.tensor(matlab_dataset['val_unseen_loc']).squeeze()-1
    test_seen_loc = torch.tensor(matlab_dataset['test_seen_loc']).squeeze()-1
    test_unseen_loc = torch.tensor(matlab_dataset['test_unseen_loc']).squeeze()-1
    species2genus = torch.tensor(matlab_dataset['species2genus'])-1
    
    dataset = DNAdataset(all_dnas, all_labels)
    dataloader= DataLoader(dataset, batch_size=32,shuffle=False)

    ###actual extraction of the feature from the model
    model.eval()
    with torch.no_grad():
        features = []
        labels = np.array([]) 
        for dnas,batch_labels in dataloader:
            #print(dnas.shape)
            dnas = dnas.to(device)
            fts = model.feature_extract(dnas)
            labels = np.concatenate((labels, batch_labels.cpu().numpy()))
            features.append(fts.cpu().numpy())
            torch.cuda.empty_cache()
        features = torch.tensor(np.concatenate(features))
        labels = torch.tensor(labels)


        expanded_train_dna_features = features[train_loc]
        expanded_train_dna_labels = labels[train_loc]
        
        expanded_val_dna_features = features[torch.cat((val_seen_loc,val_unseen_loc))]
        expanded_val_dna_labels = labels[torch.cat((val_seen_loc,val_unseen_loc))]
        expanded_test_dna_features = features[torch.cat((test_seen_loc,test_unseen_loc))]
        expanded_test_dna_labels = labels[torch.cat((test_seen_loc,test_unseen_loc))]
    return features,(expanded_train_dna_features,expanded_train_dna_labels),(expanded_val_dna_features,expanded_val_dna_labels), (expanded_test_dna_features,expanded_test_dna_labels)


