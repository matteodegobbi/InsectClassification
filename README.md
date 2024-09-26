# InsectClassification

## Main Files: 
* DnaModel.ipynb to train the CNN and extract features from DNA nucleotides
* PreTrainReACGAN.ipynb to pretrain the ReACGAN
* ReACGAN_immagini.ipynb to train the ReACGAN
* FinalClassification.ipynb for getting the final classification by combining DNA and image Features, training the InsectNet and using the top2 method to choose if a species is described or undescribed
* resnet.mlx to extract image features as the original paper
* dna_matlab.mlx to extract dna features as the original paper

---

Utils files:
* ReACGAN.py implementation of the ReACGAN, most of this code is taken from https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
* extract_features.py to extract the features from the trained GAN and trained CNN 
* dataset_utils.py used to build the .mat dataset from the raw csv and image folder

---

Some models we tried but didn't end up using are found in the directory not_used
