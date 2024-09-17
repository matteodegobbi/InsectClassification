# Insect Dataset

* all_images: vector containing the 32424 64x64x3 images (RGB) pre normalized of the insects
* all_dnas: vector containing the 32424 DNA barcodes in one-hot encoding 658x5
* all_labels: vector containing the species label for the corresponding DNA and image
* all_boldids: vector of strings containing the id from boldsystemsv3 (https://v3.boldsystems.org/) they can be used to download from boldsystems the original DNA barcodes and the full size images and other data related to the sample
* train_loc: indices of the training samples in all_dnas, all_images, all_labels, all_boldids
* val_seen_loc: indices of the validation samples in all_dnas, all_images, all_labels, all_boldids that contain described(seen) species
* val_unseen_loc: indices of the validation samples in all_dnas, all_images, all_labels, all_boldids that contain undescribed(unseen) species
* test_seen_loc: indices of the test samples in all_dnas, all_images, all_labels, all_boldids that contain described(seen) species
* test_unseen_loc: indices of the test samples in all_dnas, all_images, all_labels, all_boldids that contain undescribed(unseen) species
* species2genus: the vector contains at index i the genus label of species with label i (e.g. species i has genus species2genus[i])
* described_species_labels_train: vector containing the labels of species that appear in the training set
* described_species_labels_trainval: vector containing the labels of species that appear in the training set and/or the validation set
* all_dna_features_cnn_original: vector of features extractedfrom DNA nucleotides with the method of Badirli, S., Picard, C. J., Mohler, G.,Richert, F., Akata, Z., & Dundar, M. (2023). Classifying the
unknown: Insect identification with deep hierarchical
Bayesian learning. Methods in Ecology and Evolution, 14,
1515–1530. https://doi.org/10.1111/2041-210X.14104
* all_image_features_resnet: vector of features extracted from the insect images with the method of the same paper as the all_dna_features_cnn_original with a pretrained resnet101
* all_dna_features_cnn_new: vector of features extracted from DNA nucleotides with our CNN
* all_image_features_gan: vector of features extracted from the insect images with out method using a ReACGAN

Note: all arrays and locs are 1-indexed like in MATLAB
