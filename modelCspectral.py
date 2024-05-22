import torch
from torch import nn 
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from dataset_utils import Fit_params, Save_samples_params,save_samples
class Discriminator(torch.nn.Module):

    def __init__(self,n_feature_maps=64,n_classes=1050):
        super(Discriminator, self).__init__()
        #input 3x64x64
        self.conv1 = nn.Sequential(spectral_norm(nn.Conv2d(3,n_feature_maps,4,2,1,bias=False)),
                                   nn.LeakyReLU(0.2,inplace=True),
                                   nn.Dropout(0.5))
        self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(n_feature_maps,n_feature_maps*2,4,1,0,bias=False)),
                                   nn.BatchNorm2d(n_feature_maps*2),
                                   nn.LeakyReLU(0.2,inplace=True),
                                   nn.Dropout(0.5))
        self.conv3 = nn.Sequential(spectral_norm(nn.Conv2d(n_feature_maps*2,n_feature_maps*4,4,2,1,bias=False)),
                                   nn.BatchNorm2d(n_feature_maps*4),
                                   nn.LeakyReLU(0.2,inplace=True),
                                   nn.Dropout(0.5))
        self.conv4 = nn.Sequential(spectral_norm(nn.Conv2d(n_feature_maps*4,n_feature_maps*8,4,1,0,bias=False)),
                                   nn.BatchNorm2d(n_feature_maps*8),
                                   nn.LeakyReLU(0.2,inplace=True),
                                   nn.Dropout(0.5))
        self.conv5 = nn.Sequential(spectral_norm(nn.Conv2d(n_feature_maps*8,n_feature_maps*16,4,2,1,bias=False)),
                                   nn.BatchNorm2d(n_feature_maps*16),
                                   nn.LeakyReLU(0.2,inplace=True),
                                   nn.Dropout(0.8))
        self.conv6 = nn.Sequential(spectral_norm(nn.Conv2d(n_feature_maps*16,n_feature_maps*16,4,1,0,bias=False)),
                                   nn.BatchNorm2d(n_feature_maps*16),
                                   nn.LeakyReLU(0.2,inplace=True),
                                   nn.Dropout(0.8))
        
        self.main = nn.Sequential(self.conv1,
                                  self.conv2,
                                  self.conv3,        
                                  self.conv4,
                                  self.conv5,
                                  self.conv6)
        self.flatten = nn.Flatten() 
        self.linear_realfake = spectral_norm(nn.Linear(4096,1))
        self.linear_class = spectral_norm(nn.Linear(4096,1050))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax() 
    def forward(self, x):
        x = self.main(x)
        x = self.flatten(x)
        realfake = self.linear_realfake(x)
        realfake = self.sigmoid(realfake)
        classes = self.linear_class(x)
        classes = self.softmax(classes)
        return realfake,classes
        
class ResizeConvLayer(nn.Module):
    def __init__(self, feature_maps_in,feature_maps_out = None,kernel_size=3,padding=1):
        super().__init__()

        if feature_maps_out is None:
            feature_maps_out = feature_maps_in//2
        self.main = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                  nn.Conv2d(feature_maps_in,feature_maps_out,kernel_size,padding=padding),
                                  #nn.BatchNorm2d(feature_maps_out),
                                  nn.LeakyReLU(0.2,inplace=True),
                                 )

    def forward(self, x):
        return self.main(x)
class Tpose(nn.Module):
    def __init__(self, feature_maps_in,feature_maps_out = None,kernel_size=4, dout = 0.5):
        super().__init__()

        if feature_maps_out is None:
            feature_maps_out = feature_maps_in//2
        self.main = nn.Sequential(
            nn.ConvTranspose2d( feature_maps_in, feature_maps_out , kernel_size,2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_out),
            nn.LeakyReLU(0.2,inplace=True),
            #nn.Dropout(dout)
        )

    def forward(self, x):
        return self.main(x)
        
class Generator(nn.Module):
    def __init__(self, n_feature_maps=1024,noise_size=100,n_classes=1050,embedding_size=50):
        super(Generator, self).__init__()
        self.embed = nn.Embedding(n_classes,embedding_size)
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            
            Tpose(noise_size+embedding_size,n_feature_maps),
            Tpose(n_feature_maps),
            Tpose(n_feature_maps:=n_feature_maps//2),
            Tpose(n_feature_maps:=n_feature_maps//2),
            Tpose(n_feature_maps:=n_feature_maps//2),
            
            #ResizeConvLayer(noise_size+embedding_size,n_feature_maps),
            #ResizeConvLayer(n_feature_maps),
            #Tpose(n_feature_maps:=n_feature_maps//2),
            #ResizeConvLayer(n_feature_maps:=n_feature_maps//2),
            #ResizeConvLayer(n_feature_maps:=n_feature_maps//2),
            
            nn.ConvTranspose2d( n_feature_maps//2, 3, 4,2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, latent_noise,class_label):
        class_embedding = self.embed(class_label)
        class_embedding = class_embedding.unsqueeze(-1).unsqueeze(-1)
        concatenated_input = torch.cat((latent_noise,class_embedding),dim=1)
        out = self.main(concatenated_input)
        
        #print(out.shape)
        return out
        


#####################
#####################
#####################
#####################

from tqdm.notebook import tqdm

def train_discriminator(real_images,real_classes , discriminator_optimizer,discriminator,generator,batch_size,latent_size,described_species,device,n_classes):
    # Clear discriminator gradients
    discriminator_optimizer.zero_grad()

    # Pass real images through discriminator
    real_preds,real_predicted_classes = discriminator(real_images)
    real_preds_targets = torch.ones(real_images.size(0), 1, device=device)

    #print(real_classes)
    #print(real_predicted_classes)
    '''loss_pred_real + loss_class_real +
    loss_pred_fake + loss_class_fake'''
  
    real_loss = nn.functional.binary_cross_entropy(real_preds, real_preds_targets) + nn.functional.nll_loss(real_predicted_classes,real_classes)
    real_score = torch.mean(real_preds).item()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    #random_classes = torch.tensor(np.random.randint(0, n_classes, batch_size),device=device)
    random_classes = torch.tensor(described_species[np.random.randint(0, len(described_species), batch_size)],device=device)
    fake_images = generator(latent,random_classes)

    #print(fake_images.shape)
    # Pass fake images through discriminator
    fake_preds_targets = torch.zeros(fake_images.size(0),1, device=device)
    fake_preds,fake_predicted_classes = discriminator(fake_images)
    fake_loss = nn.functional.binary_cross_entropy(fake_preds, fake_preds_targets) + nn.functional.nll_loss(fake_predicted_classes,random_classes)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    discriminator_optimizer.step()

    class_accuracy_real = 0
    most_likely_classes = torch.argmax(real_predicted_classes, 1)
    for index in range(len(real_classes)):
        if most_likely_classes[index] == real_classes[index]:
            class_accuracy_real += 1
    class_accuracy_real /= batch_size
        
    class_accuracy_fake = 0
    most_likely_classes = torch.argmax(fake_predicted_classes, 1)
    for index in range(len(random_classes)):
        if most_likely_classes[index] == random_classes[index]:
            class_accuracy_fake += 1
    class_accuracy_fake /= batch_size
    
    return loss.item(), real_score, fake_score, class_accuracy_real, class_accuracy_fake
    
def train_generator(generator_optimizer,generator,discriminator,batch_size,latent_size,described_species,device,n_classes):
    # Clear generator gradients
    generator_optimizer.zero_grad()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1,device=device)
    #random_classes = torch.tensor(np.random.randint(0, n_classes, batch_size),device=device)
    random_classes = torch.tensor(described_species[np.random.randint(0, len(described_species), batch_size)],device=device)

    
    fake_images = generator(latent,random_classes)
    # Try to fool the discriminator
    preds,predicted_classes = discriminator(fake_images)
    preds_targets = torch.ones(batch_size, 1, device=device)
    
    loss_preds = nn.functional.binary_cross_entropy(preds, preds_targets)
    
    # target must not be one-hot encoded while predicted classes are batch_size x number_of_classes log probabilities
    loss_class = nn.functional.nll_loss(predicted_classes,random_classes)
    
    
    loss =  loss_preds + loss_class 
    # Update generator weights
    loss.backward()
    generator_optimizer.step()
    
    return loss.item()
def fit(epochs,fit_p : Fit_params, save_p : Save_samples_params,start_idx=0,latent_size =100):
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    
    for epoch in range(epochs):
        for real_images, real_classes in tqdm(fit_p.dataloaders['train']):
            # Train discriminator
            real_images = real_images.to(fit_p.device)
            real_classes = real_classes.to(fit_p.device)
            loss_d, real_score, fake_score, class_accuracy_real, class_accuracy_fake = train_discriminator(real_images, real_classes,fit_p.discriminator_optimizer,fit_p.discriminator,fit_p.generator,fit_p.batch_size,fit_p.latent_size,fit_p.described_species_labels,fit_p.device,fit_p.n_classes)
            # Train generator FOR MORE TIMES THAN DISCRIMINATOR
            #for k in range(3):
            loss_g = train_generator(fit_p.generator_optimizer,fit_p.generator,fit_p.discriminator,fit_p.batch_size,fit_p.latent_size,fit_p.described_species_labels,fit_p.device,fit_p.n_classes)
            
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        fit_p.writer.add_scalar('loss_g', loss_g, epoch)
        fit_p.writer.add_scalar('loss_d', loss_d, epoch)
        fit_p.writer.add_scalar('class_accuracy_real', class_accuracy_real, epoch)
        fit_p.writer.add_scalar('class_accuracy_fake', class_accuracy_fake, epoch)
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        print(f"class accuracy real {class_accuracy_real}")
        print(f"class accuracy fake {class_accuracy_fake}")
        # Save generated images
        save_samples(epoch+start_idx, save_p,fit_p.generator,fit_p.writer,show=False)
    
    return losses_g, losses_d, real_scores, fake_scores

