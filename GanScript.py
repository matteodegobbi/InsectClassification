import argparse
import torch
from torch import nn 
import numpy as np
import random
import dataset_utils
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
import extract_features_script
from tqdm.notebook import tqdm
import modelReACGAN as m
from utils import ops
import torch.nn.functional as F
import GanModelBuilder


def main():
    parser = argparse.ArgumentParser(description="Train or extract features from DNA")
    
    # Define arguments
    parser.add_argument("-t", "--train" ,action="store_true", help="Train the model")
    parser.add_argument("-f", "--features" ,action="store_true", help="Extract the features")
    parser.add_argument("-e","--epochs", type=int, default=12, help="Number of epochs (default: 12 )")
    parser.add_argument("-b","--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--train-on-val", action="store_true", help="Add this argument if you want to train on both training and validation set")
    parser.add_argument("--dataset-path", type=str, default="matlab_dataset/insect_dataset.mat", help="Path to the dataset for training")
    parser.add_argument("--save-weights-path", type=str, default="checkpoints/ImageGANWeights", help="Path to where to save the weights of the model after traininggt")
    parser.add_argument("--read-weights-path", type=str, default="checkpoints/ImageGANWeights", help="Path to where to read the weights of the model to extract features")

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch
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
    image_train = torch.clone(all_images[train_loc].data)
    image_val = torch.clone(torch.cat((all_images[val_seen_loc],all_images[val_unseen_loc])).data)
    image_test = torch.clone(torch.cat((all_images[test_seen_loc],all_images[test_unseen_loc])).data)
                             
    labels_train = torch.clone(all_labels[train_loc].data)
    labels_val = torch.clone(torch.cat((all_labels[val_seen_loc],all_labels[val_unseen_loc])).data)
    labels_test = torch.clone(torch.cat((all_labels[test_seen_loc],all_labels[test_unseen_loc])).data)
    
    train_d = ImageDataset(image_train,labels_train)
    val_d = ImageDataset(image_val,labels_val)
    train_val_d = ImageDataset(torch.cat((image_train,image_val)), torch.cat((labels_train,labels_val)))
    test_d = ImageDataset(image_test,labels_test)
    
    n_classes = all_labels.max()+1
    
    train_loader = torch.utils.data.DataLoader(train_d, batch_size=batch_size,shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_d, batch_size=batch_size,shuffle=True, num_workers=2)
    train_val_loader = torch.utils.data.DataLoader(train_val_d, batch_size=batch_size,shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_d, batch_size=batch_size,shuffle=True, num_workers=2)
    dataloaders = {"train":train_loader,"val":val_loader,"test":test_loader,'train_val':train_val_loader}
    dataset_sizes = {'train': len(train_d), 'val':len(val_d),'test':len(test_d),'train_val':len(train_val_d)}
    is_train_val = args.train_on_val
    if is_train_val:
        dataloaders['train'] = dataloaders['train_val']
        dataloaders['val'] = dataloaders['test']
        dataset_sizes['train'] = dataset_sizes['train_val']
        dataset_sizes['val'] = dataset_sizes['test']
        described_species_labels = described_labels_trainval
        print("Training on both train and val set")
    else:
        described_species_labels = described_labels_train

    (discriminator,generator) = GanModelBuilder.model_builder() 
    
    discriminator.to(device)
    generator.to(device)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=2e-4,betas=(0.0, 0.999))
    generator_optimizer = torch.optim.Adam(generator.parameters(),lr=2e-5,betas=(0.0, 0.999))
    
        
    cond_loss = GanModelBuilder.Data2DataCrossEntropyLoss(n_classes,0.5,0.98,device)
    cond_lambda = 1 
    
    torch.cuda.empty_cache()
    
    
    print(f"Training for {args.epochs} epochs")
    fixed_latent = torch.randn(100,100).to(device)
    import torchvision
    from tqdm.notebook import tqdm
    discriminator.train()
    generator.train()
    for epoch in range(args.epochs):
        for real_images, real_classes in tqdm(dataloaders['train']):
            real_images = real_images.to(device)
            real_classes = real_classes.to(device)
            #TRAIN DISCRIMINATOR
            for k in range(2):
                discriminator_optimizer.zero_grad()
                #use discriminator on real images
                real_dict = discriminator(real_images,real_classes)
                #use discriminator on fake images
                with torch.no_grad():
                    random_classes = torch.tensor(described_species_labels[np.random.randint(0, len(described_species_labels), batch_size)],device=device)
                    t = generator(torch.randn(batch_size,100).to(device),random_classes,eval = True)
                fake_dict = discriminator(t,random_classes)
                #Compute the two losses
                dis_acml_loss = GanModelBuilder.d_hinge(real_dict["adv_output"], fake_dict["adv_output"])
                real_cond_loss = cond_loss(**real_dict)
                dis_acml_loss += cond_lambda * real_cond_loss
                dis_acml_loss.backward()
                discriminator_optimizer.step()
    
    
            
            #TRAIN GENERATOR
            generator_optimizer.zero_grad()
            random_classes = torch.tensor(described_species_labels[np.random.randint(0, len(described_species_labels), batch_size)],device=device)
            t = generator(torch.randn(batch_size,100).to(device),random_classes,eval = True)
            fake_dict = discriminator(t,random_classes)
            gen_acml_loss = GanModelBuilder.g_hinge(fake_dict["adv_output"])
            fake_cond_loss = cond_loss(**fake_dict)
            gen_acml_loss += cond_lambda * fake_cond_loss
            gen_acml_loss.backward()
            generator_optimizer.step()
        
        print(f"disc loss={dis_acml_loss.item()}",end=',')
        print(f"gen loss={gen_acml_loss.item()}")
        with torch.no_grad():
            t = generator(fixed_latent,torch.tensor(np.arange(100)).to(device),eval = True)
        t = dataset_utils.denorm(t)
        p = torchvision.transforms.functional.to_pil_image(torchvision.utils.make_grid(t))
        p.save(f"generated/gan_training_epoch{epoch}.jpg")
            #torch.cuda.empty_cache()
            #loss_d, real_score, fake_score, class_accuracy_real, class_accuracy_fake
    
        
    print(f"Saving model weights at {args.save_weights_path}")
    torch.save({
                'epoch':args.epochs,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': generator_optimizer.state_dict(),
                }, args.save_weights_path+"_generator.pt")
    torch.save({
                'epoch': args.epochs,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': discriminator_optimizer.state_dict(),
                }, args.save_weights_path+"_discriminator.pt")
    
def feature_execution(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (discriminator,generator) = GanModelBuilder.model_builder() 
    discriminator.to(device)
    all_image_features, (tf,tl),(vf,vl),(_,_) = extract_features_script.extract_image_features(discriminator,device,args)
    features_dataset = dict()
    features_dataset['all_image_features_gan'] = all_image_features 
    io.savemat('all_image_features_gan.mat',features_dataset)
    
if __name__ == "__main__":
    main()
