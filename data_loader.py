import torch
import torchvision.datasets as dsets
from torchvision import transforms
import random
import dataset_utils
import pandas as pd 
import numpy as np
import torchvision


class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes=['church_outdoor_train','classroom_train']):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=classes, transform=transforms)
        return dataset
    
    def load_imagenet(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/imagenet', transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
        return dataset

    def load_off(self):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.ImageFolder(self.path, transform=transforms)
        return dataset

    def load_insects(self):
        transforms = self.transform(True, True, True, False)
        print(self.imsize)
        #dataset = dsets.ImageFolder(self.path, transform=transforms)
        df = pd.read_csv('final_dataset.csv',index_col=0)
        #tform = transforms.Compose([transforms.Resize((self.imsize,self.imsize)),transforms.PILToTensor(),transforms.ConvertImageDtype(torch.float),transforms.Normalize(0.5,0.5)])
        image_dataset = torchvision.datasets.ImageFolder("image_dataset/",transform=transforms)
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
        from torch.utils.data import Dataset, DataLoader
         
        class WholeDataset(Dataset):
            def __init__(self, data, transform=None):
                self.data = data
                self.targets = data.targets#torch.tensor(targets)
                #self.transform = transform
                
            def __getitem__(self, index):
                x = self.data[index][0]
                y = self.targets[index]
                
                return x, y
            
            def __len__(self):
                return len(self.data)
                
        whole_dataset = WholeDataset(image_dataset)
        n_classes = np.unique(whole_dataset.targets).shape[0]
        train_imgs = torch.utils.data.Subset(whole_dataset, train_indices)
        return train_imgs

    def load_pretrain_insects(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path, transform=transforms)
        return dataset

    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'imagenet':
            dataset = self.load_imagenet()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        elif self.dataset == 'off':
            dataset = self.load_off()
        elif self.dataset == 'insects':
            dataset = self.load_insects()
        elif self.dataset == 'pretrain_insects':
            dataset = self.load_pretrain_insects()

        print('dataset',len(dataset))
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=2,
                                              drop_last=True)
        return loader

