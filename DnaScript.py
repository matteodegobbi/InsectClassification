import argparse
import torch
from torch import nn 
import numpy as np
import random
import dataset_utils
from torch.utils.data import Dataset, DataLoader
import scipy.io as io

def main():
    parser = argparse.ArgumentParser(description="Train or extract features from DNA")
    
    # Define arguments
    parser.add_argument("-t", "--train" ,action="store_true", help="Train the model")
    parser.add_argument("-f", "--features" ,action="store_true", help="Extract the features")
    parser.add_argument("-e","--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("-b","--batch", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--train-on-val", action="store_true", help="Add this argument if you want to train on both training and validation set")
    parser.add_argument("--dataset-path", type=str, default="matlab_dataset/insect_dataset.mat", help="Path to the dataset for training")
    parser.add_argument("--save-weights-path", type=str, default="checkpoints/DnaCNNWeights.pt", help="Path to where to save the weights of the model after training")
    parser.add_argument("--read-weights-path", type=str, default="checkpoints/DnaCNNWeights.pt", help="Path to where to read the weights of the model to extract features")

    # Parse arguments
    args = parser.parse_args()

    if args.train:
        train_execution(args)
    elif args.features:
        feature_execution(args)
    else:
        parser.print_help()
        exit()


def train_execution(args):
    
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
    
    dna_train = torch.clone(all_dnas[train_loc].data)
    dna_val = torch.clone(torch.cat((all_dnas[val_seen_loc],all_dnas[val_unseen_loc])).data)
    dna_test = torch.clone(torch.cat((all_dnas[test_seen_loc],all_dnas[test_unseen_loc])).data)
    
    labels_train = torch.clone(all_labels[train_loc].data)
    labels_val = torch.clone(torch.cat((all_labels[val_seen_loc],all_labels[val_unseen_loc])).data)
    labels_test = torch.clone(torch.cat((all_labels[test_seen_loc],all_labels[test_unseen_loc])).data)
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
            
    d_train = DNAdataset(dna_train, labels_train)
    d_val = DNAdataset(dna_val, labels_val)
    d_train_val = DNAdataset(torch.cat((dna_train,dna_val)), torch.cat((labels_train,labels_val)))
    d_test = DNAdataset(dna_test,labels_test)
    
    dataloader_train = DataLoader(d_train, batch_size=args.batch,shuffle=True)
    dataloader_val = DataLoader(d_val, batch_size=args.batch,shuffle=True)
    dataloader_train_val = DataLoader(d_train_val, batch_size=args.batch,shuffle=True)
    dataloader_test = DataLoader(d_test, batch_size=args.batch,shuffle=True)
    dataloaders = {'train':dataloader_train,'val':dataloader_val,'train_val':dataloader_train_val,'test':dataloader_test}
    dataset_sizes = {'train': d_train.data.shape[0], 'val':d_val.data.shape[0],'train_val':d_train_val.data.shape[0],'test':d_test.data.shape[0]}
        
    dataloader_train = DataLoader(d_train, batch_size=args.batch,shuffle=True)
    dataloader_val = DataLoader(d_val, batch_size=args.batch,shuffle=True)
    dataloader_train_val = DataLoader(d_train_val, batch_size=args.batch,shuffle=True)
    dataloader_test = DataLoader(d_test, batch_size=args.batch,shuffle=True)
    dataloaders = {'train':dataloader_train,'val':dataloader_val,'train_val':dataloader_train_val,'test':dataloader_test}
    dataset_sizes = {'train': d_train.data.shape[0], 'val':d_val.data.shape[0],'train_val':d_train_val.data.shape[0],'test':d_test.data.shape[0]}
    
    is_train_val = args.train_on_val
    if is_train_val:
        dataloaders['train'] = dataloaders['train_val']
        dataloaders['val'] = dataloaders['test']
        dataset_sizes['train'] = dataset_sizes['train_val']
        dataset_sizes['val'] = dataset_sizes['test']
        print("Training on both train and val set")
    print(f"Training for {args.epochs} epochs")
        
    from tqdm.notebook import tqdm
    def fit(epochs,dataloaders,optimizer,model,start_idx=0):
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        
        train_losses = []
        train_scores = []
        val_losses = []
        val_scores = []
        for epoch in range(epochs):
            running_train_corrects = 0
            for dnas,labels in tqdm(dataloaders['train']):
                model.train()
                dnas = dnas.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                #print(dnas.shape)
                predicted_labels = model(dnas)
                train_loss = criterion(predicted_labels,labels)
                train_loss.backward()
                optimizer.step()
                
                _, preds = torch.max(predicted_labels, 1)
                #print(preds)
                #print(labels.data)
                running_train_corrects += torch.sum(preds == labels.data)
            train_losses.append(train_loss)
            
            running_val_corrects = 0
            for dnas,labels in tqdm(dataloaders['val']):
                
                model.eval()
                with torch.no_grad():
                    dnas = dnas.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    
                    predicted_labels = model(dnas)
                    val_loss = criterion(predicted_labels,labels)
                    
                    _, preds = torch.max(predicted_labels, 1)
                    #print(preds)
                    #print(labels.data)
                    running_val_corrects += torch.sum(preds == labels.data)
            val_losses.append(val_loss)
            
            
            
            #real_scores.append(real_score)
            #fit_p.writer.add_scalar('loss_g', loss_g, epoch)
            # Log losses & scores (last batch)
            
            epoch_train_acc = running_train_corrects.double() / dataset_sizes['train']
            epoch_val_acc = running_val_corrects.double() / dataset_sizes['val']
            print("Epoch [{}/{}], train_loss: {:.4f},  train_score: {:.4f},val_loss: {:.4f},  val_score: {:.4f}".format(
                epoch+1, epochs, train_loss, epoch_train_acc,val_loss,epoch_val_acc))
            #print(f"class accuracy real {class_accuracy_real}")
        
        return train_losses
    
    
    from DnaModel import TinyModel
    tinymodel = TinyModel()
    tinymodel.to(device)
     
    optimizer = torch.optim.Adam(tinymodel.parameters(),weight_decay=1e-5)
    n_params = dataset_utils.count_trainable_parameters(tinymodel);
    #print(n_params)
    
    fit(args.epochs,dataloaders,optimizer,tinymodel)

    print(f"Saving model weights at {args.save_weights_path}")
    torch.save({
                'epoch':args.epochs,
                'model_state_dict': tinymodel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, args.save_weights_path)
    

def feature_execution(args):
    from DnaModel import TinyModel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tinymodel = TinyModel()
    optimizer = torch.optim.Adam(tinymodel.parameters(),weight_decay=1e-5)
    tinymodel.to(device)
    state_dict = torch.load(args.read_weights_path)
    tinymodel.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    import importlib 
    import extract_features
    importlib.reload(extract_features)
    (all_dna_features,(expanded_train_dna_features,expanded_train_dna_labels),
     (expanded_val_dna_features,expanded_val_dna_labels), 
     (expanded_test_dna_features,expanded_test_dna_labels)) = \
    extract_features.extract_expanded_dna_features(tinymodel,device,
                                                   save_to_disk=False)
    
    import scipy.io as io
    features_dataset = dict()
    features_dataset['all_dna_features_cnn_new'] = all_dna_features 
    io.savemat('all_dna_features_cnn_new.mat',features_dataset)
    
if __name__ == "__main__":
    main()
